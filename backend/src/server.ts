import dotenv from 'dotenv';
import bcrypt from 'bcryptjs';
import cors from 'cors';
import express from 'express';
import helmet from 'helmet';
import jwt from 'jsonwebtoken';
import mongoose from 'mongoose';
import { OAuth2Client } from 'google-auth-library';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';

// Resolve the backend environment file relative to this module, rather than
// relying on the terminal's working directory.
dotenv.config({ path: process.env.DOTENV_CONFIG_PATH ?? fileURLToPath(new URL('../.env', import.meta.url)) });

export const app = express();
const port = Number(process.env.PORT ?? 4000);
const jwtSecret = process.env.JWT_SECRET;
const googleClientId = process.env.GOOGLE_CLIENT_ID;
const googleConfigured = Boolean(googleClientId && !/^replace-with-|^YOUR_CLIENT_ID|^your[-_]/i.test(googleClientId));
const googleClient = googleConfigured ? new OAuth2Client(googleClientId) : null;
let databaseAvailable = false;
let databaseState: 'not-configured' | 'connecting' | 'connected' | 'error' = process.env.MONGODB_URI ? 'connecting' : 'not-configured';
let databaseError = '';
app.use(helmet());
const allowedOrigins = (process.env.FRONTEND_ORIGIN ?? 'http://localhost:5173').split(',').map(origin => origin.trim()).filter(Boolean);
app.use(cors({ origin: (origin, callback) => callback(null, !origin || allowedOrigins.includes(origin)) }));
// DenseNet state dictionaries are large even though they contain no image data. Keep
// this independently configurable so the privacy boundary does not accidentally make
// legitimate federated updates impossible to submit.
app.use(express.json({ limit: process.env.MODEL_UPDATE_MAX_BYTES ?? '200mb', type: ['application/json', 'application/*+json'] }));
if (process.env.MONGODB_URI && process.env.NODE_ENV !== 'test') {
  mongoose.connect(process.env.MONGODB_URI, { dbName: process.env.MONGODB_DB ?? 'arogyavaani', serverSelectionTimeoutMS: 5000 })
    .then(() => { databaseAvailable = true; databaseState = 'connected'; databaseError = ''; void ensureAdmin(); console.log('MongoDB connected'); })
    .catch(error => { databaseState = 'error'; databaseError = error instanceof Error ? error.message : 'MongoDB connection failed'; console.warn(`MongoDB unavailable: ${databaseError}`); });
  mongoose.connection.on('disconnected', () => { databaseAvailable = false; databaseState = 'error'; databaseError = 'MongoDB disconnected'; });
}

type Role = 'hospital' | 'admin';
type Parameters = Record<string, number[]>;
const loose = new mongoose.Schema({}, { strict: false, _id: false });
const User = mongoose.models.User ?? mongoose.model('User', new mongoose.Schema({
  email: { type: String, required: true, unique: true, lowercase: true, trim: true, index: true },
  hospitalName: String, role: { type: String, enum: ['hospital', 'admin'], required: true, index: true },
  passwordHash: { type: String, select: false }, googleSubject: { type: String, unique: true, sparse: true, index: true },
}, { timestamps: true }));
const ModelUpdate = mongoose.models.ModelUpdate ?? mongoose.model('ModelUpdate', new mongoose.Schema({
  hospitalId: { type: String, required: true, index: true }, round: { type: Number, required: true, index: true }, roundId: { type: String, required: true, index: true },
  samples: { type: Number, required: true }, classes: [{ type: String, required: true }], architecture: { type: String, required: true }, baseVersion: { type: Number, required: true },
  binaryPayload: { type: Buffer, required: true }, parameterShapes: loose, metrics: loose, submittedAt: { type: Date, default: Date.now },
}, { timestamps: true }).index({ round: 1, hospitalId: 1 }, { unique: true }));
const FederatedRound = mongoose.models.FederatedRound ?? mongoose.model('FederatedRound', new mongoose.Schema({
  round: { type: Number, required: true, unique: true, index: true }, baseGlobalModelVersion: { type: Number, required: true },
  participantHospitalIds: { type: [String], required: true }, submittedHospitalIds: { type: [String], default: [] },
  expectedParticipantCount: { type: Number, required: true }, submittedParticipantCount: { type: Number, default: 0 }, totalSamples: { type: Number, default: 0 },
  status: { type: String, enum: ['DRAFT', 'ACTIVE', 'WAITING_FOR_UPDATES', 'READY_FOR_AGGREGATION', 'AGGREGATING', 'COMPLETED', 'CANCELLED', 'CLOSED'], required: true, index: true },
  startedAt: Date, readyAt: Date, aggregationStartedAt: Date, completedAt: Date, resultingGlobalModelVersion: Number, aggregationError: String,
}, { timestamps: true }).index({ status: 1, round: -1 }));
const GlobalModel = mongoose.models.GlobalModel ?? mongoose.model('GlobalModel', new mongoose.Schema({
  version: { type: Number, unique: true, index: true }, round: Number, samples: Number, hospitals: Number, classes: [String], architecture: String, parameterShapes: loose, parameters: { type: loose }, binaryPayload: Buffer, status: String,
}, { timestamps: true }));
const Prediction = mongoose.models.Prediction ?? mongoose.model('Prediction', new mongoose.Schema({
  hospitalId: { type: String, required: true, index: true }, modelVersion: Number, predictedClass: String, confidence: Number, timestamp: { type: Date, default: Date.now },
}, { timestamps: true }));
const TrainingRun = mongoose.models.TrainingRun ?? mongoose.model('TrainingRun', new mongoose.Schema({
  hospitalId: { type: String, required: true, index: true }, status: { type: String, required: true, index: true }, samples: Number,
  epochs: Number, loss: Number, accuracy: Number, checkpoint: String, message: String, startedAt: Date, completedAt: Date,
}, { timestamps: true }).index({ hospitalId: 1, createdAt: -1 }));
const AuditLog = mongoose.models.AuditLog ?? mongoose.model('AuditLog', new mongoose.Schema({
  actorId: { type: String, required: true, index: true }, actorRole: { type: String, enum: ['hospital', 'admin'], required: true },
  action: { type: String, required: true, index: true }, target: String, details: loose,
}, { timestamps: true }).index({ createdAt: -1 }));

const memoryUsers = new Map<string, { email: string; hospitalName?: string; role: Role; passwordHash?: string; googleSubject?: string }>();
type RoundStatus = 'DRAFT' | 'ACTIVE' | 'WAITING_FOR_UPDATES' | 'READY_FOR_AGGREGATION' | 'AGGREGATING' | 'COMPLETED' | 'CANCELLED' | 'CLOSED';
type MemoryRound = { _id: string; round: number; baseGlobalModelVersion: number; participantHospitalIds: string[]; submittedHospitalIds: string[]; expectedParticipantCount: number; submittedParticipantCount: number; totalSamples: number; status: RoundStatus; createdAt: Date; startedAt?: Date; readyAt?: Date; aggregationStartedAt?: Date; completedAt?: Date; resultingGlobalModelVersion?: number; aggregationError?: string };
type MemoryUpdate = { hospitalId: string; round: number; roundId: string; samples: number; binaryPayload: Buffer; classes: string[]; architecture: string; baseVersion: number; parameterShapes: Record<string, unknown>; metrics?: Record<string, unknown>; submittedAt: Date };
let memoryUpdates: MemoryUpdate[] = [];
let memoryModels: any[] = [];
let memoryRounds: MemoryRound[] = [];
let memoryTrainingRuns: Array<Record<string, unknown>> = [];
let memoryBinaryModels: any[] = [];
let memoryAuditLogs: Array<Record<string, unknown>> = [];
const useDb = () => databaseAvailable && mongoose.connection.readyState === 1;
const adminEmail = process.env.ADMIN_EMAIL?.trim().toLowerCase();
const adminPassword = process.env.ADMIN_PASSWORD;
// The administrator is provisioned only from server-side configuration.  This
// also means there is no public path that can create an administrator account.
async function ensureAdmin() {
  if (!adminEmail || !adminPassword) return;
  if (useDb()) {
    const existing: any = await User.findOne({ email: adminEmail }).select('+passwordHash').lean();
    if (!existing || existing.role !== 'admin' || !existing.passwordHash || !await bcrypt.compare(adminPassword, existing.passwordHash)) {
      await User.findOneAndUpdate({ email: adminEmail }, { email: adminEmail, role: 'admin', passwordHash: await bcrypt.hash(adminPassword, 12), hospitalName: undefined, googleSubject: undefined }, { upsert: true, new: true, setDefaultsOnInsert: true });
    }
  } else {
    const existing = memoryUsers.get(adminEmail);
    if (!existing || existing.role !== 'admin' || !existing.passwordHash || !await bcrypt.compare(adminPassword, existing.passwordHash)) memoryUsers.set(adminEmail, { email: adminEmail, role: 'admin', passwordHash: await bcrypt.hash(adminPassword, 12) });
  }
}
const currentModel = async (): Promise<any> => useDb() ? GlobalModel.findOne().sort({ version: -1 }).lean() : [...memoryModels, ...memoryBinaryModels].sort((left, right) => right.version - left.version)[0];
const currentRound = async (): Promise<any> => useDb() ? FederatedRound.findOne().sort({ round: -1 }).lean() : memoryRounds.at(-1);
const recordAudit = async (actorId: string, actorRole: Role, action: string, target?: string, details?: Record<string, unknown>) => {
  const entry = { actorId, actorRole, action, target, details, createdAt: new Date() };
  if (useDb()) await AuditLog.create(entry); else memoryAuditLogs.push(entry);
};
const modelMetadata = (model: any) => model && ({ version: model.version, round: model.round, samples: model.samples, hospitals: model.hospitals, classes: model.classes, architecture: model.architecture, parameterShapes: model.parameterShapes, status: model.status, createdAt: model.createdAt, updatedAt: model.updatedAt });
const updateMetadata = (update: any) => ({ hospitalId: update.hospitalId, round: update.round, samples: update.samples, classes: update.classes, architecture: update.architecture, baseVersion: update.baseVersion, parameterShapes: update.parameterShapes, metrics: update.metrics, receivedAt: update.submittedAt, createdAt: update.createdAt, payloadBytes: update.binaryPayload?.length });
type AuthenticatedRequest = express.Request & { user?: { sub: string; role: Role } };
const auth = (role?: Role) => (req: AuthenticatedRequest, res: express.Response, next: express.NextFunction) => {
  const token = req.headers.authorization?.startsWith('Bearer ') ? req.headers.authorization.slice(7) : undefined;
  if (!token || !jwtSecret) return res.status(401).json({ message: 'Authentication required.' });
  try {
    const user = jwt.verify(token, jwtSecret) as { sub: string; role: Role };
    if (!user.sub || (role && user.role !== role)) return res.status(403).json({ message: 'Insufficient permissions.' });
    req.user = user; next();
  } catch { return res.status(401).json({ message: 'Your session has expired. Please sign in again.' }); }
};
const issueToken = (email: string, role: Role) => jwt.sign({ sub: email, role }, jwtSecret!, { expiresIn: '8h' });
const compatibleKeys = (updates: Array<{ parameters: Parameters }>) => {
  const keys = Object.keys(updates[0].parameters).sort();
  if (!keys.length || updates.some(update => Object.keys(update.parameters).sort().join('|') !== keys.join('|') || keys.some(key => update.parameters[key].length !== updates[0].parameters[key].length))) throw new Error('Model update tensor shapes do not match.');
  return keys;
};
const sameParameterShapes = (left: Record<string, unknown> | undefined, right: Record<string, unknown>) => {
  const keys = Object.keys(left ?? {}).sort();
  return keys.join('|') === Object.keys(right).sort().join('|') && keys.every(key => JSON.stringify(left?.[key]) === JSON.stringify(right[key]));
};
type WireTensor = { name: string; shape: number[]; dtype: 'f32' | 'f64' | 'i64' | 'i32' | 'u8' | 'bool'; offset: number; length: number };
type WireManifest = { format: string; classes: string[]; architecture: string; baseVersion: number; tensors: WireTensor[] };
type WireModel = { manifest: WireManifest; dataOffset: number; payload: Buffer };
const wireBytes: Record<WireTensor['dtype'], number> = { f32: 4, f64: 8, i64: 8, i32: 4, u8: 1, bool: 1 };
const parseWireModel = (payload: Buffer): WireModel => {
  if (payload.length < 8 || payload.subarray(0, 4).toString('ascii') !== 'AVM1') throw new Error('Invalid binary model payload.');
  const manifestSize = payload.readUInt32LE(4); const dataOffset = 8 + manifestSize;
  if (manifestSize > 2_000_000 || payload.length < dataOffset) throw new Error('Invalid binary model manifest.');
  const rawManifest = JSON.parse(payload.subarray(8, dataOffset).toString('utf8')) as Partial<WireManifest>;
  if (rawManifest.format !== 'arogyavaani-state-v1' || !Array.isArray(rawManifest.classes) || !rawManifest.classes.every(item => typeof item === 'string' && item.length) || !Array.isArray(rawManifest.tensors)) throw new Error('Unsupported binary model payload.');
  const manifest: WireManifest = { format: rawManifest.format, classes: rawManifest.classes, architecture: typeof rawManifest.architecture === 'string' && rawManifest.architecture.length <= 128 ? rawManifest.architecture : 'densenet121', baseVersion: Number.isInteger(rawManifest.baseVersion) && (rawManifest.baseVersion as number) >= 0 ? rawManifest.baseVersion as number : 0, tensors: rawManifest.tensors as WireTensor[] };
  const names = new Set<string>();
  for (const tensor of manifest.tensors) {
    const elements = Array.isArray(tensor.shape) ? tensor.shape.reduce((total, dimension) => Number.isInteger(dimension) && dimension > 0 ? total * dimension : NaN, 1) : NaN;
    if (!tensor || typeof tensor.name !== 'string' || !tensor.name || names.has(tensor.name) || !wireBytes[tensor.dtype] || !Number.isInteger(tensor.offset) || !Number.isInteger(tensor.length) || !Number.isFinite(elements) || elements * wireBytes[tensor.dtype] !== tensor.length || tensor.offset < 0 || tensor.length < 0 || tensor.offset + tensor.length > payload.length - dataOffset) throw new Error('Invalid binary tensor manifest.');
    names.add(tensor.name);
  }
  return { manifest, dataOffset, payload };
};
const validateWireValues = (wire: WireModel) => {
  for (const tensor of wire.manifest.tensors) {
    if (tensor.dtype === 'f32') for (let index = 0; index < tensor.length; index += 4) if (!Number.isFinite(wire.payload.readFloatLE(wire.dataOffset + tensor.offset + index))) throw new Error('Model update contains invalid floating-point parameters.');
    if (tensor.dtype === 'f64') for (let index = 0; index < tensor.length; index += 8) if (!Number.isFinite(wire.payload.readDoubleLE(wire.dataOffset + tensor.offset + index))) throw new Error('Model update contains invalid floating-point parameters.');
  }
};
const aggregateWireModels = (updates: Array<{ binaryPayload: Buffer; samples: number }>) => {
  const models = updates.map(item => ({ ...item, model: parseWireModel(item.binaryPayload) })); const first = models[0].model;
  const signature = (model: WireModel) => model.manifest.architecture + '|' + model.manifest.tensors.map(t => t.name + ':' + t.dtype + ':' + t.shape.join('x') + ':' + t.length).join('|');
  if (models.some(item => signature(item.model) !== signature(first))) throw new Error('Model update tensor shapes do not match.');
  if (models.some(item => item.model.manifest.classes.join('|') !== first.manifest.classes.join('|'))) throw new Error('Model class definitions do not match.');
  const totalSamples = models.reduce((sum, item) => sum + item.samples, 0); const chunks: Buffer[] = []; const tensors: WireTensor[] = []; let offset = 0;
  for (const tensor of first.manifest.tensors) {
    const chunk = Buffer.alloc(tensor.length);
    if (tensor.dtype === 'f32') for (let index = 0; index < tensor.length; index += 4) { const value = models.reduce((sum, item) => sum + item.model.payload.readFloatLE(item.model.dataOffset + tensor.offset + index) * item.samples, 0) / totalSamples; if (!Number.isFinite(value)) throw new Error('Model update contains invalid floating-point parameters.'); chunk.writeFloatLE(value, index); }
    else if (tensor.dtype === 'f64') for (let index = 0; index < tensor.length; index += 8) { const value = models.reduce((sum, item) => sum + item.model.payload.readDoubleLE(item.model.dataOffset + tensor.offset + index) * item.samples, 0) / totalSamples; if (!Number.isFinite(value)) throw new Error('Model update contains invalid floating-point parameters.'); chunk.writeDoubleLE(value, index); }
    // Integer/bool buffers in a PyTorch state dictionary (for example batch
    // counters) are not averaged; retain the compatible reference value.
    else models[0].model.payload.copy(chunk, 0, first.dataOffset + tensor.offset, first.dataOffset + tensor.offset + tensor.length);
    tensors.push({ ...tensor, offset }); offset += chunk.length; chunks.push(chunk);
  }
  const manifest = Buffer.from(JSON.stringify({ format: 'arogyavaani-state-v1', classes: first.manifest.classes, architecture: first.manifest.architecture, baseVersion: first.manifest.baseVersion, tensors }));
  const size = Buffer.alloc(4); size.writeUInt32LE(manifest.length);
  return { binaryPayload: Buffer.concat([Buffer.from('AVM1'), size, manifest, ...chunks]), classes: first.manifest.classes, architecture: first.manifest.architecture, parameterShapes: Object.fromEntries(first.manifest.tensors.map(t => [t.name, t.shape])), totalSamples };
};

app.get('/api/health', (_req, res) => res.status(databaseState === 'error' ? 503 : 200).json({ status: databaseState === 'error' ? 'degraded' : 'operational', persistence: useDb() ? 'mongodb' : databaseState, database: { configured: Boolean(process.env.MONGODB_URI), state: databaseState, error: process.env.NODE_ENV !== 'production' ? databaseError || undefined : undefined }, privacyBoundary: 'model-updates-and-metadata-only' }));
app.post('/api/auth/register', async (req, res) => {
  const parsed = z.object({ hospitalName: z.string().min(2), email: z.string().email(), password: z.string().min(12) }).strict().safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ message: 'Please provide a hospital name, a valid email address, and a password of at least 12 characters.' });
  const data = { ...parsed.data, email: parsed.data.email.toLowerCase() };
  const existing = useDb() ? await User.exists({ email: data.email }) : memoryUsers.has(data.email);
  if (existing) return res.status(409).json({ message: 'An account already exists for this email.' });
  const passwordHash = await bcrypt.hash(data.password, 12);
  if (useDb()) await User.create({ ...data, passwordHash, role: 'hospital' }); else memoryUsers.set(data.email, { ...data, passwordHash, role: 'hospital' });
  await recordAudit(data.email, 'hospital', 'hospital.registered');
  res.status(201).json({ message: 'Hospital account created. You can now sign in.' });
});
app.post('/api/auth/login', async (req, res) => {
  const parsed = z.object({ email: z.string().email(), password: z.string().min(1) }).strict().safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ message: 'Enter a valid email address and password.' });
  if (!jwtSecret) return res.status(503).json({ message: 'Sign-in is temporarily unavailable. Please try again later.' });
  const email = parsed.data.email.toLowerCase();
  await ensureAdmin();
  const dbUser: any = useDb() ? await User.findOne({ email }).select('+passwordHash').lean() : memoryUsers.get(email);
  const valid = Boolean(dbUser?.passwordHash && await bcrypt.compare(parsed.data.password, dbUser.passwordHash));
  if (!valid) return res.status(401).json({ message: 'Invalid credentials.' });
  await recordAudit(email, dbUser.role, 'auth.signed_in');
  res.json({ token: issueToken(email, dbUser.role), role: dbUser.role });
});
app.get('/api/auth/session', auth(), async (req: AuthenticatedRequest, res) => {
  const user: any = useDb() ? await User.findOne({ email: req.user!.sub }).lean() : memoryUsers.get(req.user!.sub);
  res.json({ email: req.user!.sub, role: req.user!.role, hospitalName: user?.hospitalName });
});
app.post('/api/auth/logout', auth(), async (req: AuthenticatedRequest, res) => {
  await recordAudit(req.user!.sub, req.user!.role, 'auth.signed_out');
  res.status(204).end();
});
// OAuth client IDs identify the public browser application; they are not client
// secrets. Returning the configured ID lets a deployment use one backend .env
// file while the backend still performs the actual ID-token verification.
app.get('/api/auth/google/status', (_req, res) => res.json({ configured: googleConfigured, clientId: googleConfigured ? googleClientId : undefined }));
app.post('/api/auth/google', async (req, res) => {
  const parsed = z.object({ idToken: z.string().min(1), hospitalName: z.string().min(2).optional() }).strict().safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ message: 'A valid Google ID token is required.' });
  if (!googleClient || !googleClientId || !googleConfigured || !jwtSecret) return res.status(503).json({ message: 'Google sign-in is unavailable right now. Please use email and password.' });
  try {
    const ticket = await googleClient.verifyIdToken({ idToken: parsed.data.idToken, audience: googleClientId });
    const payload = ticket.getPayload();
    if (!payload?.sub || !payload.email || !payload.email_verified) return res.status(401).json({ message: 'Google did not provide a verified email address.' });
    if (!['accounts.google.com', 'https://accounts.google.com'].includes(payload.iss ?? '')) return res.status(401).json({ message: 'Google token issuer is invalid.' });
    const email = payload.email.toLowerCase();
    const existing: any = useDb() ? await User.findOne({ $or: [{ googleSubject: payload.sub }, { email }] }).select('+passwordHash').lean() : memoryUsers.get(email);
    if (existing) {
      if (existing.googleSubject && existing.googleSubject !== payload.sub) return res.status(409).json({ message: 'This email is already linked to a different Google account.' });
      if (!existing.googleSubject) return res.status(409).json({ message: 'An account already exists for this email. Sign in with its existing method.' });
      await recordAudit(email, existing.role, 'auth.google_signed_in');
      return res.json({ token: issueToken(email, existing.role), role: existing.role });
    }
    if (!parsed.data.hospitalName) return res.status(400).json({ message: 'Please provide your hospital name to finish registration.' });
    const user = { email, role: 'hospital' as const, googleSubject: payload.sub, hospitalName: parsed.data.hospitalName };
    if (useDb()) await User.create(user); else memoryUsers.set(email, user);
    await recordAudit(email, user.role, 'auth.google_onboarded');
    res.status(201).json({ token: issueToken(email, user.role), role: user.role, onboarding: true });
  } catch { return res.status(401).json({ message: 'Google ID token could not be verified.' }); }
});
// A round freezes its participant list and base model.  Registered hospitals are
// deliberately not used as a proxy for participants.
const openStatuses: RoundStatus[] = ['DRAFT', 'ACTIVE', 'WAITING_FOR_UPDATES', 'READY_FOR_AGGREGATION', 'AGGREGATING'];
const findRound = async (round: number): Promise<any> => useDb() ? FederatedRound.findOne({ round }).lean() : memoryRounds.find(item => item.round === round);
const activeRound = async (): Promise<any> => useDb() ? FederatedRound.findOne({ status: { $in: openStatuses } }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => openStatuses.includes(item.status));
const roundView = (round: any, hospitalId?: string) => round && ({ id: String(round._id), round: round.round, status: round.status, baseGlobalModelVersion: round.baseGlobalModelVersion, participantHospitalIds: round.participantHospitalIds, submittedHospitalIds: round.submittedHospitalIds, expectedParticipantCount: round.expectedParticipantCount, submittedParticipantCount: round.submittedParticipantCount, remainingParticipantCount: Math.max(0, round.expectedParticipantCount - round.submittedParticipantCount), totalSamples: round.totalSamples, resultingGlobalModelVersion: round.resultingGlobalModelVersion, createdAt: round.createdAt, startedAt: round.startedAt, readyAt: round.readyAt, aggregationStartedAt: round.aggregationStartedAt, completedAt: round.completedAt, aggregationError: round.aggregationError, isParticipant: hospitalId ? round.participantHospitalIds.includes(hospitalId) : undefined, hasSubmitted: hospitalId ? round.submittedHospitalIds.includes(hospitalId) : undefined });

async function aggregateRound(roundNumber: number) {
  const claimed: any = useDb() ? await FederatedRound.findOneAndUpdate({ round: roundNumber, status: 'READY_FOR_AGGREGATION' }, { $set: { status: 'AGGREGATING', aggregationStartedAt: new Date(), aggregationError: undefined } }, { new: true }).lean() : (() => { const item = memoryRounds.find(round => round.round === roundNumber && round.status === 'READY_FOR_AGGREGATION'); if (item) { item.status = 'AGGREGATING'; item.aggregationStartedAt = new Date(); item.aggregationError = undefined; } return item; })();
  if (!claimed) return;
  await recordAudit('system', 'admin', 'federated.aggregation_started', `round:${roundNumber}`);
  try {
    const updates: any[] = useDb() ? await ModelUpdate.find({ round: roundNumber }).lean() : memoryUpdates.filter(item => item.round === roundNumber);
    if (updates.length !== claimed.expectedParticipantCount || updates.some(update => !claimed.participantHospitalIds.includes(update.hospitalId))) throw new Error('Round does not contain every required participant update.');
    const latest = await currentModel();
    if ((latest?.version ?? 0) !== claimed.baseGlobalModelVersion || updates.some(update => update.baseVersion !== claimed.baseGlobalModelVersion)) throw new Error('One or more updates were trained from an outdated global model.');
    const aggregate = aggregateWireModels(updates);
    const result = { version: claimed.baseGlobalModelVersion + 1, round: roundNumber, samples: aggregate.totalSamples, hospitals: updates.length, binaryPayload: aggregate.binaryPayload, classes: aggregate.classes, architecture: aggregate.architecture, parameterShapes: aggregate.parameterShapes, status: 'Ready to synchronize', createdAt: new Date() };
    if (useDb()) { await GlobalModel.create(result); await FederatedRound.updateOne({ _id: claimed._id, status: 'AGGREGATING' }, { $set: { status: 'COMPLETED', resultingGlobalModelVersion: result.version, completedAt: new Date(), totalSamples: result.samples } }); }
    else { memoryBinaryModels.push(result); const item = memoryRounds.find(round => round.round === roundNumber)!; item.status = 'COMPLETED'; item.resultingGlobalModelVersion = result.version; item.completedAt = new Date(); item.totalSamples = result.samples; }
    await recordAudit('system', 'admin', 'federated.aggregation_completed', `round:${roundNumber}`, { version: result.version, hospitals: updates.length, samples: result.samples });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Aggregation failed.';
    if (useDb()) await FederatedRound.updateOne({ _id: claimed._id, status: 'AGGREGATING' }, { $set: { status: 'READY_FOR_AGGREGATION', aggregationError: message } }); else { const item = memoryRounds.find(round => round.round === roundNumber)!; item.status = 'READY_FOR_AGGREGATION'; item.aggregationError = message; }
    await recordAudit('system', 'admin', 'federated.aggregation_failed', `round:${roundNumber}`, { message });
  }
}
app.post('/api/federated/rounds', auth('admin'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ participantHospitalIds: z.array(z.string().email()).min(1).max(500) }).strict().safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ message: 'Select at least one valid registered hospital.' });
  if (await activeRound()) return res.status(409).json({ message: 'Close or complete the current federated round before creating another.' });
  const participants = [...new Set(parsed.data.participantHospitalIds.map(item => item.toLowerCase()))];
  const registered = useDb() ? (await User.find({ email: { $in: participants }, role: 'hospital' }).select('email').lean()).map(item => item.email) : participants.filter(item => memoryUsers.get(item)?.role === 'hospital');
  if (registered.length !== participants.length) return res.status(400).json({ message: 'Every selected participant must be a registered hospital.' });
  const previous = await currentRound(); const model = await currentModel(); const now = new Date(); const round: MemoryRound = { _id: new mongoose.Types.ObjectId().toString(), round: (previous?.round ?? 0) + 1, baseGlobalModelVersion: model?.version ?? 0, participantHospitalIds: participants, submittedHospitalIds: [], expectedParticipantCount: participants.length, submittedParticipantCount: 0, totalSamples: 0, status: 'ACTIVE', createdAt: now, startedAt: now };
  const saved: any = useDb() ? await FederatedRound.create(round) : (memoryRounds.push(round), round);
  await recordAudit(req.user!.sub, 'admin', 'federated.round_created', `round:${round.round}`, { participants: participants.length, baseGlobalModelVersion: round.baseGlobalModelVersion });
  res.status(201).json(roundView(saved));
});
app.get('/api/federated/rounds', auth('admin'), async (_req, res) => res.json(((useDb() ? await FederatedRound.find().sort({ round: -1 }).lean() : [...memoryRounds].reverse()) as any[]).map(round => roundView(round))));
app.get('/api/federated/rounds/active', auth(), async (req: AuthenticatedRequest, res) => {
  const round = await activeRound() ?? (req.user!.role === 'hospital' ? (useDb() ? await FederatedRound.findOne({ participantHospitalIds: req.user!.sub }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => item.participantHospitalIds.includes(req.user!.sub))) : undefined);
  if (!round) return res.status(404).json({ message: 'No active federated round is available.' });
  res.json(roundView(round, req.user!.role === 'hospital' ? req.user!.sub : undefined));
});
app.get('/api/federated/rounds/:round', auth(), async (req: AuthenticatedRequest, res) => { const round = await findRound(Number(req.params.round)); if (!round) return res.status(404).json({ message: 'Federated round not found.' }); if (req.user!.role === 'hospital' && !round.participantHospitalIds.includes(req.user!.sub)) return res.status(403).json({ message: 'You are not a participant in this round.' }); res.json(roundView(round, req.user!.role === 'hospital' ? req.user!.sub : undefined)); });
app.post('/api/federated/rounds/:round/cancel', auth('admin'), async (req: AuthenticatedRequest, res) => { const round = await findRound(Number(req.params.round)); if (!round) return res.status(404).json({ message: 'Federated round not found.' }); if (!['DRAFT', 'ACTIVE', 'WAITING_FOR_UPDATES', 'READY_FOR_AGGREGATION'].includes(round.status)) return res.status(409).json({ message: 'Only an incomplete round can be cancelled.' }); if (useDb()) await FederatedRound.updateOne({ _id: round._id, status: round.status }, { $set: { status: 'CANCELLED', completedAt: new Date() } }); else { round.status = 'CANCELLED'; round.completedAt = new Date(); } await recordAudit(req.user!.sub, 'admin', 'federated.round_cancelled', `round:${round.round}`); res.json(roundView(await findRound(round.round))); });
async function submitRoundUpdate(req: AuthenticatedRequest, res: express.Response, roundNumber: number, samples: number) {
  if (!Buffer.isBuffer(req.body) || !req.body.length) return res.status(400).json({ message: 'An AVM1 model payload is required.' });
  const round = await findRound(roundNumber); if (!round) return res.status(404).json({ message: 'Federated round not found.' });
  if (!['ACTIVE', 'WAITING_FOR_UPDATES'].includes(round.status)) return res.status(409).json({ message: `Round is ${round.status} and cannot accept updates.` });
  if (!round.participantHospitalIds.includes(req.user!.sub)) return res.status(403).json({ message: 'This hospital is not selected for the round.' });
  if (round.submittedHospitalIds.includes(req.user!.sub)) return res.status(409).json({ message: 'This hospital has already submitted an update for this round.' });
  try {
    const wire = parseWireModel(req.body); validateWireValues(wire); const parameterShapes = Object.fromEntries(wire.manifest.tensors.map(tensor => [tensor.name, tensor.shape]));
    if (wire.manifest.baseVersion !== round.baseGlobalModelVersion) return res.status(409).json({ message: `This round requires global model v${round.baseGlobalModelVersion}. Synchronize and train again.` });
    const model = await currentModel(); if (model && (model.version !== round.baseGlobalModelVersion || model.architecture !== wire.manifest.architecture || model.classes?.join('|') !== wire.manifest.classes.join('|') || !sameParameterShapes(model.parameterShapes, parameterShapes))) return res.status(409).json({ message: 'The update is incompatible with the required global model.' });
    const update: MemoryUpdate = { hospitalId: req.user!.sub, round: roundNumber, roundId: String(round._id), samples, binaryPayload: req.body, classes: wire.manifest.classes, architecture: wire.manifest.architecture, baseVersion: wire.manifest.baseVersion, parameterShapes, submittedAt: new Date() };
    try { if (useDb()) await ModelUpdate.create(update); else { if (memoryUpdates.some(item => item.round === roundNumber && item.hospitalId === update.hospitalId)) throw new Error('duplicate'); memoryUpdates.push(update); } } catch { return res.status(409).json({ message: 'This hospital has already submitted an update for this round.' }); }
    let updated: any;
    if (useDb()) updated = await FederatedRound.findOneAndUpdate({ _id: round._id, status: { $in: ['ACTIVE', 'WAITING_FOR_UPDATES'] }, submittedHospitalIds: { $ne: req.user!.sub } }, { $addToSet: { submittedHospitalIds: req.user!.sub }, $inc: { submittedParticipantCount: 1, totalSamples: samples }, $set: { status: 'WAITING_FOR_UPDATES' } }, { new: true }).lean(); else { const item = memoryRounds.find(value => value.round === roundNumber)!; if (!item.submittedHospitalIds.includes(req.user!.sub)) { item.submittedHospitalIds.push(req.user!.sub); item.submittedParticipantCount++; item.totalSamples += samples; item.status = 'WAITING_FOR_UPDATES'; updated = item; } }
    if (!updated) { if (useDb()) await ModelUpdate.deleteOne({ round: roundNumber, hospitalId: req.user!.sub }); else memoryUpdates = memoryUpdates.filter(item => item !== update); return res.status(409).json({ message: 'Round state changed; update was not accepted.' }); }
    if (updated.submittedParticipantCount === updated.expectedParticipantCount) { if (useDb()) updated = await FederatedRound.findOneAndUpdate({ _id: updated._id, status: 'WAITING_FOR_UPDATES' }, { $set: { status: 'READY_FOR_AGGREGATION', readyAt: new Date() } }, { new: true }).lean() ?? updated; else { updated.status = 'READY_FOR_AGGREGATION'; updated.readyAt = new Date(); } void aggregateRound(roundNumber); }
    await recordAudit(req.user!.sub, 'hospital', 'federated.update_submitted', `round:${roundNumber}`, { samples, payloadBytes: req.body.length });
    res.status(202).json({ accepted: true, bytes: req.body.length, message: updated.status === 'READY_FOR_AGGREGATION' ? 'Final required update received; aggregation has started.' : 'Model update stored. Waiting for other participating hospitals.', ...roundView(updated, req.user!.sub) });
  } catch (error) { await recordAudit(req.user!.sub, 'hospital', 'federated.invalid_update_rejected', `round:${roundNumber}`); res.status(400).json({ message: error instanceof Error ? error.message : 'Invalid binary model payload.' }); }
}
app.post('/api/federated/rounds/:round/updates', express.raw({ type: 'application/vnd.arogyavaani.model-v1', limit: process.env.MODEL_UPDATE_MAX_BYTES ?? '200mb' }), auth('hospital'), async (req: AuthenticatedRequest, res) => { const parsed = z.object({ round: z.coerce.number().int().positive(), samples: z.coerce.number().int().positive() }).safeParse({ round: req.params.round, samples: req.query.samples }); if (!parsed.success) return res.status(400).json({ message: 'A positive sample count is required.' }); await submitRoundUpdate(req, res, parsed.data.round, parsed.data.samples); });
app.post('/api/federated/update-binary', express.raw({ type: 'application/vnd.arogyavaani.model-v1', limit: process.env.MODEL_UPDATE_MAX_BYTES ?? '200mb' }), auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ samples: z.coerce.number().int().positive(), round: z.coerce.number().int().positive() }).safeParse(req.query);
  if (!parsed.success) return res.status(400).json({ message: 'A positive round and sample count are required.' });
  await submitRoundUpdate(req, res, parsed.data.round, parsed.data.samples);
});
app.post('/api/federated/update', auth('hospital'), async (req: AuthenticatedRequest, res) => {
  res.status(410).json({ message: 'JSON parameter updates are retired. Submit an AVM1 payload to the selected federated round.' });
});
app.get('/api/federated/status', auth(), async (req: AuthenticatedRequest, res) => {
  const model = await currentModel(); const round: any = await activeRound() ?? (req.user!.role === 'hospital' ? (useDb() ? await FederatedRound.findOne({ participantHospitalIds: req.user!.sub }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => item.participantHospitalIds.includes(req.user!.sub))) : undefined);
  res.json({ version: model?.version ?? 0, modelStatus: model?.status ?? 'No model available', architecture: model?.architecture, round: round?.round ?? 0, samples: model?.samples ?? 0, hospitals: round?.expectedParticipantCount ?? 0, updatesReceived: round?.submittedParticipantCount ?? 0, aggregationStatus: round?.status ?? 'NO_ACTIVE_ROUND', updatedAt: model?.updatedAt ?? model?.createdAt ?? new Date(), baseGlobalModelVersion: round?.baseGlobalModelVersion, isParticipant: round && req.user!.role === 'hospital' ? round.participantHospitalIds.includes(req.user!.sub) : undefined });
});
app.post('/api/federated/aggregate', auth('admin'), async (_req, res) => res.status(410).json({ message: 'Aggregation is automatic and waits for every selected participant.' }));
app.post('/api/federated/aggregate-binary', auth('admin'), async (_req, res) => res.status(410).json({ message: 'Aggregation is automatic and waits for every selected participant.' }));
app.post('/api/federated/rounds/:round/aggregate', auth('admin'), async (req, res) => { const round = await findRound(Number(req.params.round)); if (!round) return res.status(404).json({ message: 'Federated round not found.' }); if (round.status !== 'READY_FOR_AGGREGATION') return res.status(409).json({ message: 'Aggregation requires every selected participant update.' }); void aggregateRound(round.round); res.status(202).json({ message: 'Aggregation is running.' }); });
app.get('/api/federated/global-model', auth('hospital'), async (_req, res) => { const model = await currentModel(); if (!model) return res.status(404).json({ message: 'No global model has been aggregated yet.' }); res.json(model); });
app.get('/api/federated/global-model-binary', auth('hospital'), async (req, res) => {
  const version = z.coerce.number().int().nonnegative().safeParse(req.query.version);
  const model: any = useDb() ? await GlobalModel.findOne(version.success ? { version: version.data, binaryPayload: { $exists: true } } : { binaryPayload: { $exists: true } }).sort({ version: -1 }).lean() : (version.success ? memoryBinaryModels.find(item => item.version === version.data) : memoryBinaryModels.at(-1));
  if (!model?.binaryPayload) return res.status(404).json({ message: 'No binary global model has been aggregated yet.' });
  res.setHeader('Content-Type', 'application/vnd.arogyavaani.model-v1'); res.setHeader('Content-Length', model.binaryPayload.length); res.setHeader('X-ArogyaVaani-Model-Version', String(model.version)); res.send(model.binaryPayload);
});
app.post('/api/training/runs', auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ status: z.enum(['queued', 'validating', 'training', 'completed', 'failed']), samples: z.number().int().nonnegative().optional(), epochs: z.number().int().positive().optional(), loss: z.number().finite().optional(), accuracy: z.number().finite().optional(), checkpoint: z.string().max(512).optional(), message: z.string().max(1000).optional(), startedAt: z.string().datetime().optional(), completedAt: z.string().datetime().optional() }).safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ message: 'Invalid training run metadata.' });
  const run = { ...parsed.data, hospitalId: req.user!.sub, startedAt: parsed.data.startedAt ? new Date(parsed.data.startedAt) : undefined, completedAt: parsed.data.completedAt ? new Date(parsed.data.completedAt) : undefined };
  if (useDb()) await TrainingRun.create(run); else memoryTrainingRuns.push({ ...run, createdAt: new Date() });
  await recordAudit(req.user!.sub, 'hospital', 'training.run_recorded', undefined, { status: run.status, samples: run.samples });
  res.status(201).json({ recorded: true });
});
app.get('/api/federated/history', auth(), async (_req, res) => res.json(useDb() ? await FederatedRound.find().sort({ round: -1 }).lean() : [...memoryRounds].reverse()));
app.get('/api/training/runs', auth('hospital'), async (req: AuthenticatedRequest, res) => res.json(useDb() ? await TrainingRun.find({ hospitalId: req.user!.sub }).sort({ createdAt: -1 }).lean() : memoryTrainingRuns.filter(run => run.hospitalId === req.user!.sub).reverse()));
app.get('/api/hospital/overview', auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const user: any = useDb() ? await User.findOne({ email: req.user!.sub }).lean() : memoryUsers.get(req.user!.sub);
  const model = await currentModel(); const round: any = await activeRound() ?? (useDb() ? await FederatedRound.findOne({ participantHospitalIds: req.user!.sub }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => item.participantHospitalIds.includes(req.user!.sub)));
  const lastUpdate: any = useDb() ? await ModelUpdate.findOne({ hospitalId: req.user!.sub }).sort({ submittedAt: -1 }).lean() : memoryUpdates.filter(update => update.hospitalId === req.user!.sub).sort((a, b) => b.submittedAt.getTime() - a.submittedAt.getTime())[0];
  const lastTraining: any = useDb() ? await TrainingRun.findOne({ hospitalId: req.user!.sub }).sort({ createdAt: -1 }).lean() : memoryTrainingRuns.filter(run => run.hospitalId === req.user!.sub).at(-1);
  res.json({ hospitalName: user?.hospitalName ?? 'Hospital', connectionStatus: databaseState === 'error' ? 'Degraded' : 'Connected', globalModel: model ? { version: model.version, round: model.round, status: model.status, updatedAt: model.updatedAt ?? model.createdAt } : null, currentRound: round?.round ?? 0, round: roundView(round, req.user!.sub), training: lastTraining ? { status: lastTraining.status, samples: lastTraining.samples, loss: lastTraining.loss, accuracy: lastTraining.accuracy, updatedAt: lastTraining.completedAt ?? lastTraining.createdAt } : { status: 'No local training recorded' }, participationStatus: round ? (round.submittedHospitalIds.includes(req.user!.sub) ? 'Update submitted for this round' : round.participantHospitalIds.includes(req.user!.sub) ? 'Awaiting local update' : 'Not selected for this round') : 'No active round', lastContribution: lastUpdate ? { round: lastUpdate.round, samples: lastUpdate.samples, receivedAt: lastUpdate.submittedAt } : null });
});
app.get('/api/admin/dashboard', auth('admin'), async (_req, res) => { const hospitals = useDb() ? await User.countDocuments({ role: 'hospital' }) : [...memoryUsers.values()].filter(user => user.role === 'hospital').length; const model = await currentModel(); const round: any = await activeRound(); res.json({ totalHospitals: hospitals, activeHospitals: round?.submittedParticipantCount ?? 0, currentRound: round?.round ?? 0, globalVersion: model?.version ?? 0, updatesReceived: round?.submittedParticipantCount ?? 0, activeRound: roundView(round), systemHealth: 'Operational' }); });
app.get('/api/admin/hospitals', auth('admin'), async (_req, res) => res.json(useDb() ? await User.find({ role: 'hospital' }).select('email hospitalName createdAt').lean() : [...memoryUsers.values()].filter(user => user.role === 'hospital').map(({ passwordHash, ...user }) => user)));
app.get('/api/admin/federated-rounds', auth('admin'), async (_req, res) => res.json(useDb() ? await FederatedRound.find().sort({ round: -1 }).lean() : [...memoryRounds].reverse()));
app.get('/api/admin/models', auth('admin'), async (_req, res) => res.json((useDb() ? await GlobalModel.find().sort({ version: -1 }).lean() : [...memoryModels, ...memoryBinaryModels].sort((left, right) => right.version - left.version)).map(modelMetadata)));
app.get('/api/admin/model-updates', auth('admin'), async (_req, res) => res.json(((useDb() ? await ModelUpdate.find().sort({ submittedAt: -1 }).lean() : [...memoryUpdates].sort((left, right) => right.submittedAt.getTime() - left.submittedAt.getTime())) as any[]).map(updateMetadata)));
app.get('/api/admin/audit-logs', auth('admin'), async (_req, res) => res.json(useDb() ? await AuditLog.find().sort({ createdAt: -1 }).limit(100).lean() : [...memoryAuditLogs].reverse().slice(0, 100)));
app.post('/api/predictions/metadata', auth('hospital'), async (req: AuthenticatedRequest, res) => { const parsed = z.object({ modelVersion: z.number().int().nonnegative(), predictedClass: z.string().min(1), confidence: z.number().min(0).max(1) }).safeParse(req.body); if (!parsed.success) return res.status(400).json({ message: 'Only prediction metadata may be recorded.' }); if (useDb()) await Prediction.create({ ...parsed.data, hospitalId: req.user!.sub }); await recordAudit(req.user!.sub, 'hospital', 'prediction.metadata_recorded', undefined, { modelVersion: parsed.data.modelVersion }); res.status(201).json({ recorded: true }); });
if (process.env.NODE_ENV !== 'test') app.listen(port, () => console.log(`ArogyaVaani API listening on http://localhost:${port}`));
