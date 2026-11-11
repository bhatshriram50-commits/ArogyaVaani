import dotenv from 'dotenv';
import bcrypt from 'bcryptjs';
import cors from 'cors';
import express from 'express';
import helmet from 'helmet';
import jwt from 'jsonwebtoken';
import mongoose from 'mongoose';
import { OAuth2Client } from 'google-auth-library';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';
import { enabledTaskFor, medicalTasks, taskFor, type MedicalTaskId } from './medicalTasks.js';

// Resolve the backend environment file relative to this module, rather than
// relying on the terminal's working directory.
dotenv.config({ path: process.env.DOTENV_CONFIG_PATH ?? fileURLToPath(new URL('../.env', import.meta.url)) });

export const app = express();
const port = Number(process.env.PORT ?? 4000);
const jwtSecret = process.env.JWT_SECRET;
const production = process.env.NODE_ENV === 'production';
const persistenceRequired = production;
const artifactRoot = path.resolve(process.env.MODEL_ARTIFACT_ROOT ?? fileURLToPath(new URL('../model_artifacts', import.meta.url)));
const rawFrontendOrigin = process.env.FRONTEND_ORIGIN?.trim();
const isLocalhost = (origin: string) => /^https?:\/\/(localhost|127\.0\.0\.1)(?::\d+)?$/i.test(origin);
const configuredFrontendOrigins = (rawFrontendOrigin ? rawFrontendOrigin.split(',') : ['http://localhost:5173'])
  .map(origin => origin.trim().replace(/\/+$/, ''))
  .filter(Boolean)
  .map(origin => {
    if (!/^https?:\/\//i.test(origin) && origin !== '*' && !isLocalhost(`http://${origin}`)) {
      return `https://${origin}`;
    }
    return origin;
  });

if (production) {
  const configurationErrors = [
    !jwtSecret || jwtSecret.length < 32 ? 'JWT_SECRET must be at least 32 characters' : '',
    !process.env.MONGODB_URI ? 'MONGODB_URI is required' : '',
    !process.env.ADMIN_EMAIL || !process.env.ADMIN_PASSWORD ? 'ADMIN_EMAIL and ADMIN_PASSWORD are required' : '',
    !artifactRoot ? 'MODEL_ARTIFACT_ROOT is required' : '',
    rawFrontendOrigin && configuredFrontendOrigins.some(origin => origin !== '*' && !origin.startsWith('https://') && !isLocalhost(origin))
      ? 'FRONTEND_ORIGIN must contain HTTPS or localhost origins'
      : '',
  ].filter(Boolean);
  if (configurationErrors.length) throw new Error(`Unsafe production configuration: ${configurationErrors.join('; ')}`);
  if (!rawFrontendOrigin) {
    console.warn('WARN: FRONTEND_ORIGIN is not set in production. Set FRONTEND_ORIGIN to your HTTPS Vercel URL in your dashboard.');
  }
}
const googleClientId = process.env.GOOGLE_CLIENT_ID;
const googleConfigured = Boolean(googleClientId && !/^replace-with-|^YOUR_CLIENT_ID|^your[-_]/i.test(googleClientId));
const googleClient = googleConfigured ? new OAuth2Client(googleClientId) : null;
let databaseAvailable = false;
let databaseState: 'not-configured' | 'connecting' | 'connected' | 'error' = process.env.MONGODB_URI ? 'connecting' : 'not-configured';
let databaseError = '';
app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' }
}));
const allowedOrigins = configuredFrontendOrigins;
const localDevelopmentOrigin = (origin: string) => isLocalhost(origin);
app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (!rawFrontendOrigin || allowedOrigins.includes('*')) return callback(null, true);
    if (allowedOrigins.includes(origin) || localDevelopmentOrigin(origin)) return callback(null, true);
    callback(null, false);
  }
}));
// DenseNet state dictionaries are large even though they contain no image data. Keep
// this independently configurable so the privacy boundary does not accidentally make
// legitimate federated updates impossible to submit.
app.use(express.json({ limit: process.env.MODEL_UPDATE_MAX_BYTES ?? '200mb', type: ['application/json', 'application/*+json'] }));
type RateWindow = { count: number; resetAt: number };
const rateWindows = new Map<string, RateWindow>();
const rateLimit = (bucket: string, max: number, windowMs = 15 * 60_000) => (req: express.Request, res: express.Response, next: express.NextFunction) => {
  const key = `${bucket}:${req.ip}`; const now = Date.now(); const previous = rateWindows.get(key);
  const state = !previous || previous.resetAt <= now ? { count: 0, resetAt: now + windowMs } : previous;
  state.count += 1; rateWindows.set(key, state);
  if (state.count > max) return res.status(429).json({ message: 'Too many requests. Please try again later.' });
  next();
};
app.use((req, res, next) => {
  if (persistenceRequired && !useDb() && !['/api/health', '/api/ready'].includes(req.path)) return res.status(503).json({ message: 'Persistent storage is unavailable. The API is not accepting requests.' });
  const started = Date.now();
  res.on('finish', () => { if (res.statusCode >= 400) console.warn(JSON.stringify({ event: 'API_REQUEST', method: req.method, path: req.path, status: res.statusCode, durationMs: Date.now() - started })); });
  next();
});
let gridFsBucket: mongoose.mongo.GridFSBucket | undefined;
if (process.env.MONGODB_URI && process.env.NODE_ENV !== 'test') {
  mongoose.connect(process.env.MONGODB_URI, { dbName: process.env.MONGODB_DB ?? 'arogyavaani', serverSelectionTimeoutMS: 5000 })
    .then(async () => {
      if (mongoose.connection.db) {
        gridFsBucket = new mongoose.mongo.GridFSBucket(mongoose.connection.db, { bucketName: 'model_artifacts' });
      }
      if (artifactRoot) await mkdir(artifactRoot, { recursive: true });
      await migrateLegacyPneumoniaData();
      databaseAvailable = true;
      databaseState = 'connected';
      databaseError = '';
      void ensureAdmin();
      console.log(JSON.stringify({ event: 'DATABASE_CONNECTED' }));
    })
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
  agentLastSeen: Date, agentTasks: [String],
}, { timestamps: true }));
const ModelUpdate = mongoose.models.ModelUpdate ?? mongoose.model('ModelUpdate', new mongoose.Schema({
  task: { type: String, required: true, default: 'PNEUMONIA', index: true }, hospitalId: { type: String, required: true, index: true }, round: { type: Number, required: true, index: true }, roundId: { type: String, required: true, index: true },
  samples: { type: Number, required: true }, classes: [{ type: String, required: true }], architecture: { type: String, required: true }, baseVersion: { type: Number, required: true },
  // Large AVM1 blobs are stored as artifacts in persistent storage. binaryPayload
  // remains optional solely to read pre-existing development/legacy records.
  binaryPayload: Buffer, artifactKey: String, checksum: String, payloadBytes: Number, parameterShapes: loose, metrics: loose, submittedAt: { type: Date, default: Date.now },
}, { timestamps: true }).index({ task: 1, round: 1, hospitalId: 1 }, { unique: true }));
const FederatedRound = mongoose.models.FederatedRound ?? mongoose.model('FederatedRound', new mongoose.Schema({
  task: { type: String, required: true, default: 'PNEUMONIA', index: true }, round: { type: Number, required: true, index: true }, baseGlobalModelVersion: { type: Number, required: true },
  participantHospitalIds: { type: [String], required: true }, submittedHospitalIds: { type: [String], default: [] },
  expectedParticipantCount: { type: Number, required: true }, submittedParticipantCount: { type: Number, default: 0 }, totalSamples: { type: Number, default: 0 },
  status: { type: String, enum: ['DRAFT', 'ACTIVE', 'WAITING_FOR_UPDATES', 'READY_FOR_AGGREGATION', 'AGGREGATING', 'COMPLETED', 'CANCELLED', 'CLOSED'], required: true, index: true },
  startedAt: Date, readyAt: Date, aggregationStartedAt: Date, completedAt: Date, resultingGlobalModelVersion: Number, aggregationError: String,
}, { timestamps: true }).index({ task: 1, round: 1 }, { unique: true }).index({ task: 1, status: 1, round: -1 }));
const GlobalModel = mongoose.models.GlobalModel ?? mongoose.model('GlobalModel', new mongoose.Schema({
  task: { type: String, required: true, default: 'PNEUMONIA', index: true }, version: { type: Number, required: true }, round: Number, samples: Number, hospitals: Number, classes: [String], architecture: String, parameterShapes: loose, parameters: { type: loose }, binaryPayload: Buffer, artifactKey: String, checksum: String, status: String,
}, { timestamps: true }).index({ task: 1, version: 1 }, { unique: true }));
const Prediction = mongoose.models.Prediction ?? mongoose.model('Prediction', new mongoose.Schema({
  hospitalId: { type: String, required: true, index: true }, task: { type: String, required: true, default: 'PNEUMONIA', index: true }, modelVersion: Number, predictedClass: String, confidence: Number, timestamp: { type: Date, default: Date.now },
}, { timestamps: true }));
const TrainingRun = mongoose.models.TrainingRun ?? mongoose.model('TrainingRun', new mongoose.Schema({
  hospitalId: { type: String, required: true, index: true }, task: { type: String, required: true, default: 'PNEUMONIA', index: true }, status: { type: String, required: true, index: true }, samples: Number,
  epochs: Number, loss: Number, accuracy: Number, checkpoint: String, message: String, startedAt: Date, completedAt: Date,
}, { timestamps: true }).index({ hospitalId: 1, createdAt: -1 }));
const AuditLog = mongoose.models.AuditLog ?? mongoose.model('AuditLog', new mongoose.Schema({
  actorId: { type: String, required: true, index: true }, actorRole: { type: String, enum: ['hospital', 'admin'], required: true },
  action: { type: String, required: true, index: true }, target: String, details: loose,
}, { timestamps: true }).index({ createdAt: -1 }));

// The pre-task product supported Pneumonia only.  Those untagged documents
// therefore have an unambiguous identity; any separately introduced legacy
// data must be explicitly tagged by an operator rather than guessed here.
async function migrateLegacyPneumoniaData() {
  const missingTask = { $or: [{ task: { $exists: false } }, { task: null }] };
  await Promise.all([
    ModelUpdate.updateMany(missingTask, { $set: { task: 'PNEUMONIA' } }),
    FederatedRound.updateMany(missingTask, { $set: { task: 'PNEUMONIA' } }),
    GlobalModel.updateMany(missingTask, { $set: { task: 'PNEUMONIA' } }),
    Prediction.updateMany(missingTask, { $set: { task: 'PNEUMONIA' } }),
    TrainingRun.updateMany(missingTask, { $set: { task: 'PNEUMONIA' } }),
  ]);
  // MongoDB retains old indexes after a Mongoose schema changes.  Remove only
  // the former global uniqueness constraints so (task, round/version) can be
  // independent; all other indexes are left untouched.
  for (const [collection, name] of [[FederatedRound.collection, 'round_1'], [ModelUpdate.collection, 'round_1_hospitalId_1'], [GlobalModel.collection, 'version_1']] as const) {
    const indexes = await collection.indexes();
    if (indexes.some(index => index.name === name)) await collection.dropIndex(name);
  }
}

const memoryUsers = new Map<string, { email: string; hospitalName?: string; role: Role; passwordHash?: string; googleSubject?: string; agentLastSeen?: Date; agentTasks?: string[] }>();
type RoundStatus = 'DRAFT' | 'ACTIVE' | 'WAITING_FOR_UPDATES' | 'READY_FOR_AGGREGATION' | 'AGGREGATING' | 'COMPLETED' | 'CANCELLED' | 'CLOSED';
type MemoryRound = { _id: string; task: MedicalTaskId; round: number; baseGlobalModelVersion: number; participantHospitalIds: string[]; submittedHospitalIds: string[]; expectedParticipantCount: number; submittedParticipantCount: number; totalSamples: number; status: RoundStatus; createdAt: Date; startedAt?: Date; readyAt?: Date; aggregationStartedAt?: Date; completedAt?: Date; resultingGlobalModelVersion?: number; aggregationError?: string };
type MemoryUpdate = { task: MedicalTaskId; hospitalId: string; round: number; roundId: string; samples: number; binaryPayload: Buffer; classes: string[]; architecture: string; baseVersion: number; parameterShapes: Record<string, unknown>; metrics?: Record<string, unknown>; submittedAt: Date };
let memoryUpdates: MemoryUpdate[] = [];
let memoryModels: any[] = [];
let memoryRounds: MemoryRound[] = [];
let memoryTrainingRuns: Array<Record<string, unknown>> = [];
let memoryBinaryModels: any[] = [];
let memoryAuditLogs: Array<Record<string, unknown>> = [];
const useDb = () => databaseAvailable && mongoose.connection.readyState === 1;
const checksum = (payload: Buffer) => createHash('sha256').update(payload).digest('hex');
const artifactKeyFor = (kind: 'updates' | 'models', task: MedicalTaskId) => `${kind}/${task.toLowerCase()}/${randomUUID()}.avm1`;
// Model blobs deliberately stay out of normal MongoDB documents. This small
// boundary is provider-neutral: an S3/Azure/GCS implementation need only
// implement these methods, leaving federation and authorization unchanged.
interface ModelArtifactStore {
  save(key: string, payload: Buffer): Promise<void>;
  read(key: string, expectedChecksum?: string): Promise<Buffer>;
  remove(key: string): Promise<void>;
}
class FilesystemArtifactStore implements ModelArtifactStore {
  constructor(private readonly root: string) {}
  private pathFor(key: string) {
    if (!/^(updates|models)\/[a-z_]+\/[0-9a-f-]+\.avm1$/.test(key)) throw new Error('Invalid model artifact key.');
    const candidate = path.resolve(this.root, key);
    if (!candidate.startsWith(this.root + path.sep)) throw new Error('Invalid model artifact key.');
    return candidate;
  }
  async save(key: string, payload: Buffer) {
    const destination = this.pathFor(key); await mkdir(path.dirname(destination), { recursive: true });
    const temporary = `${destination}.${randomUUID()}.tmp`;
    await writeFile(temporary, payload, { flag: 'wx' }); await rename(temporary, destination);
  }
  async read(key: string, expectedChecksum?: string) {
    const payload = await readFile(this.pathFor(key));
    if (expectedChecksum && checksum(payload) !== expectedChecksum) throw new Error('Model artifact integrity verification failed.');
    return payload;
  }
  async remove(key: string) { await unlink(this.pathFor(key)).catch(() => undefined); }
}
class MongoGridFSArtifactStore implements ModelArtifactStore {
  constructor(private readonly getBucket: () => mongoose.mongo.GridFSBucket | undefined, private readonly fallbackFs?: FilesystemArtifactStore) {}

  async save(key: string, payload: Buffer) {
    const normalizedKey = key.replace(/\\/g, '/');
    const bucket = this.getBucket();
    if (bucket) {
      try {
        const existing = await bucket.find({ filename: normalizedKey }).toArray();
        for (const file of existing) await bucket.delete(file._id).catch(() => undefined);
        const uploadStream = bucket.openUploadStream(normalizedKey, { chunkSizeBytes: 1048576 });
        await new Promise<void>((resolve, reject) => {
          uploadStream.on('finish', () => resolve());
          uploadStream.on('error', reject);
          uploadStream.end(payload);
        });
      } catch (err) {
        console.warn('GridFS save warning:', err instanceof Error ? err.message : err);
      }
    }
    if (this.fallbackFs) {
      await this.fallbackFs.save(key, payload).catch(err => console.warn('Filesystem fallback save warning:', err));
    }
  }

  async read(key: string, expectedChecksum?: string) {
    const normalizedKey = key.replace(/\\/g, '/');
    const bucket = this.getBucket();
    if (bucket) {
      try {
        const existing = await bucket.find({ $or: [{ filename: normalizedKey }, { filename: key }] }).toArray();
        if (existing.length > 0) {
          const chunks: Buffer[] = [];
          const downloadStream = bucket.openDownloadStreamByName(existing[0].filename);
          for await (const chunk of downloadStream) {
            chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
          }
          const payload = Buffer.concat(chunks);
          if (expectedChecksum && checksum(payload) !== expectedChecksum) {
            throw new Error('Model artifact integrity verification failed.');
          }
          return payload;
        }
      } catch (err) {
        console.warn('GridFS read warning, trying filesystem:', err instanceof Error ? err.message : err);
      }
    }
    if (this.fallbackFs) {
      return this.fallbackFs.read(key, expectedChecksum);
    }
    throw new Error('Model artifact not found in GridFS or filesystem: ' + key);
  }

  async remove(key: string) {
    const normalizedKey = key.replace(/\\/g, '/');
    const bucket = this.getBucket();
    if (bucket) {
      try {
        const existing = await bucket.find({ $or: [{ filename: normalizedKey }, { filename: key }] }).toArray();
        for (const file of existing) await bucket.delete(file._id).catch(() => undefined);
      } catch { /* ignore */ }
    }
    if (this.fallbackFs) {
      await this.fallbackFs.remove(key).catch(() => undefined);
    }
  }
}
const fsArtifactStore = artifactRoot ? new FilesystemArtifactStore(artifactRoot) : undefined;
const artifactStore: ModelArtifactStore = new MongoGridFSArtifactStore(() => gridFsBucket, fsArtifactStore);
const saveArtifact = async (key: string, payload: Buffer) => {
  await artifactStore.save(key, payload);
};
const readArtifact = async (key: string, expectedChecksum?: string) => {
  return artifactStore.read(key, expectedChecksum);
};
const removeArtifact = async (key?: string) => { if (key) await artifactStore.remove(key); };
const payloadFor = async (record: any) => {
  const payload = record.binaryPayload ? Buffer.from(record.binaryPayload) : record.artifactKey ? await readArtifact(record.artifactKey, record.checksum) : undefined;
  if (!payload) throw new Error('Model artifact is unavailable.');
  return payload;
};
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
// A task has one explicitly active model.  Legacy records predate the active
// marker, so they continue to resolve to their newest version until the first
// new aggregation or rollback marks a model active.
const currentModel = async (task: MedicalTaskId = 'PNEUMONIA'): Promise<any> => {
  if (useDb()) return await GlobalModel.findOne({ task, status: 'ACTIVE' }).sort({ version: -1 }).lean() ?? await GlobalModel.findOne({ task }).sort({ version: -1 }).lean();
  const models = [...memoryModels, ...memoryBinaryModels].filter(model => (model.task ?? 'PNEUMONIA') === task);
  return models.filter(model => model.status === 'ACTIVE').sort((left, right) => right.version - left.version)[0] ?? models.sort((left, right) => right.version - left.version)[0];
};
const currentRound = async (task: MedicalTaskId = 'PNEUMONIA'): Promise<any> => useDb() ? FederatedRound.findOne({ task }).sort({ round: -1 }).lean() : memoryRounds.filter(round => round.task === task).at(-1);
const recordAudit = async (actorId: string, actorRole: Role, action: string, target?: string, details?: Record<string, unknown>) => {
  const entry = { actorId, actorRole, action, target, details, createdAt: new Date() };
  if (useDb()) await AuditLog.create(entry); else memoryAuditLogs.push(entry);
};
const modelMetadata = (model: any) => model && ({ task: model.task ?? 'PNEUMONIA', version: model.version, round: model.round, samples: model.samples, hospitals: model.hospitals, classes: model.classes, architecture: model.architecture, parameterShapes: model.parameterShapes, checksum: model.checksum, status: model.status, createdAt: model.createdAt, updatedAt: model.updatedAt });
const updateMetadata = (update: any) => ({ task: update.task ?? 'PNEUMONIA', hospitalId: update.hospitalId, round: update.round, samples: update.samples, classes: update.classes, architecture: update.architecture, baseVersion: update.baseVersion, parameterShapes: update.parameterShapes, metrics: update.metrics, receivedAt: update.submittedAt, createdAt: update.createdAt, payloadBytes: update.payloadBytes ?? update.binaryPayload?.length, checksum: update.checksum, status: 'stored' });
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
const flattenShapes = (obj: Record<string, unknown> | undefined, prefix = ''): Record<string, number[]> => {
  if (!obj || typeof obj !== 'object') return {};
  const result: Record<string, number[]> = {};
  for (const [key, value] of Object.entries(obj)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    if (Array.isArray(value) && value.every(v => typeof v === 'number')) {
      result[fullKey] = value;
    } else if (value && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(result, flattenShapes(value as Record<string, unknown>, fullKey));
    }
  }
  return result;
};
const sameParameterShapes = (left: Record<string, unknown> | undefined, right: Record<string, unknown>) => {
  if (!left || !right) return false;
  const flatLeft = flattenShapes(left);
  const flatRight = flattenShapes(right);
  const leftKeys = Object.keys(flatLeft).sort();
  const rightKeys = Object.keys(flatRight).sort();
  if (!leftKeys.length || !rightKeys.length) return true;
  return leftKeys.join('|') === rightKeys.join('|') && leftKeys.every(key => JSON.stringify(flatLeft[key]) === JSON.stringify(flatRight[key]));
};
type WireTensor = { name: string; shape: number[]; dtype: 'f32' | 'f64' | 'i64' | 'i32' | 'u8' | 'bool'; offset: number; length: number };
type WireManifest = { format: string; task?: MedicalTaskId; classes: string[]; architecture: string; baseVersion: number; tensors: WireTensor[] };
type WireModel = { manifest: WireManifest; dataOffset: number; payload: Buffer };
const wireBytes: Record<WireTensor['dtype'], number> = { f32: 4, f64: 8, i64: 8, i32: 4, u8: 1, bool: 1 };
const parseWireModel = (payload: Buffer): WireModel => {
  if (payload.length < 8 || payload.subarray(0, 4).toString('ascii') !== 'AVM1') throw new Error('Invalid binary model payload.');
  const manifestSize = payload.readUInt32LE(4); const dataOffset = 8 + manifestSize;
  if (manifestSize > 2_000_000 || payload.length < dataOffset) throw new Error('Invalid binary model manifest.');
  const rawManifest = JSON.parse(payload.subarray(8, dataOffset).toString('utf8')) as Partial<WireManifest>;
  if (rawManifest.format !== 'arogyavaani-state-v1' || !Array.isArray(rawManifest.classes) || rawManifest.classes.length < 2 || rawManifest.classes.length > 1_000 || !rawManifest.classes.every(item => typeof item === 'string' && item.length <= 256) || !Array.isArray(rawManifest.tensors) || rawManifest.tensors.length > 20_000) throw new Error('Unsupported binary model payload.');
  if (rawManifest.task !== undefined && !taskFor(rawManifest.task)) throw new Error('The binary model payload has an invalid medical task identity.');
  const manifest: WireManifest = { format: rawManifest.format, task: taskFor(rawManifest.task)?.taskId, classes: rawManifest.classes, architecture: typeof rawManifest.architecture === 'string' && rawManifest.architecture.length <= 128 ? rawManifest.architecture : 'densenet121', baseVersion: Number.isInteger(rawManifest.baseVersion) && (rawManifest.baseVersion as number) >= 0 ? rawManifest.baseVersion as number : 0, tensors: rawManifest.tensors as WireTensor[] };
  const names = new Set<string>();
  for (const tensor of manifest.tensors) {
    const elements = Array.isArray(tensor.shape) ? tensor.shape.reduce((total, dimension) => Number.isInteger(dimension) && dimension > 0 && dimension <= 100_000_000 ? total * dimension : NaN, 1) : NaN;
    if (!tensor || typeof tensor.name !== 'string' || !tensor.name || tensor.name.length > 512 || names.has(tensor.name) || !wireBytes[tensor.dtype] || !Number.isInteger(tensor.offset) || !Number.isInteger(tensor.length) || !Number.isFinite(elements) || elements > 100_000_000 || elements * wireBytes[tensor.dtype] !== tensor.length || tensor.offset < 0 || tensor.length < 0 || tensor.offset + tensor.length > payload.length - dataOffset) throw new Error('Invalid binary tensor manifest.');
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
const aggregateWireModels = (updates: Array<{ binaryPayload: Buffer; samples: number }>, resultingVersion?: number) => {
  const models = updates.map(item => ({ ...item, model: parseWireModel(item.binaryPayload) })); const first = models[0].model;
  const signature = (model: WireModel) => (model.manifest.task ?? 'PNEUMONIA') + '|' + model.manifest.architecture + '|' + model.manifest.tensors.map(t => t.name + ':' + t.dtype + ':' + t.shape.join('x') + ':' + t.length).join('|');
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
  const manifest = Buffer.from(JSON.stringify({ format: 'arogyavaani-state-v1', task: first.manifest.task, classes: first.manifest.classes, architecture: first.manifest.architecture, baseVersion: resultingVersion ?? first.manifest.baseVersion, tensors }));
  const size = Buffer.alloc(4); size.writeUInt32LE(manifest.length);
  return { binaryPayload: Buffer.concat([Buffer.from('AVM1'), size, manifest, ...chunks]), classes: first.manifest.classes, architecture: first.manifest.architecture, parameterShapes: Object.fromEntries(first.manifest.tensors.map(t => [t.name, t.shape])), totalSamples };
};

app.get('/', (_req, res) => res.json({ name: 'ArogyaVaani Central API', status: 'operational', version: '1.0.0', health: '/api/health', ready: '/api/ready' }));
app.get('/api/health', (_req, res) => res.status(databaseState === 'error' ? 503 : 200).json({ status: databaseState === 'error' ? 'degraded' : 'operational', persistence: useDb() ? 'mongodb' : databaseState, database: { configured: Boolean(process.env.MONGODB_URI), state: databaseState, error: process.env.NODE_ENV !== 'production' ? databaseError || undefined : undefined }, privacyBoundary: 'model-updates-and-metadata-only' }));
app.get('/api/ready', (_req, res) => {
  const ready = persistenceRequired ? useDb() && Boolean(artifactRoot) : databaseState !== 'error';
  res.status(ready ? 200 : 503).json({ status: ready ? 'ready' : 'not-ready', persistence: useDb() ? 'mongodb' : databaseState, artifactStorage: artifactRoot ? 'configured' : 'not-configured' });
});
app.get('/api/medical-tasks', auth(), (_req, res) => res.json(Object.values(medicalTasks).filter(task => task.enabled)));
app.get('/api/medical-tasks/:task', auth(), (req, res) => { const task = enabledTaskFor(req.params.task); if (!task) return res.status(404).json({ message: 'Medical task is unavailable or disabled.' }); res.json(task); });
app.post('/api/auth/register', rateLimit('register', 20), async (req, res) => {
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
app.post('/api/auth/login', rateLimit('login', 30), async (req, res) => {
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
app.post('/api/auth/google', rateLimit('google-auth', 20), async (req, res) => {
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
const findRound = async (round: number, task: MedicalTaskId = 'PNEUMONIA'): Promise<any> => useDb() ? FederatedRound.findOne({ task, round }).lean() : memoryRounds.find(item => item.task === task && item.round === round);
const activeRound = async (task: MedicalTaskId = 'PNEUMONIA'): Promise<any> => useDb() ? FederatedRound.findOne({ task, status: { $in: openStatuses } }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => item.task === task && openStatuses.includes(item.status));
// Passing a hospital id produces the deliberately reduced hospital view.  The
// participant roster and every other hospital's submission state are central
// operational data and must never be returned to a hospital browser.
const roundView = (round: any, hospitalId?: string) => round && ({ id: String(round._id), task: round.task ?? 'PNEUMONIA', round: round.round, roundNumber: round.round, status: round.status, baseGlobalModelVersion: round.baseGlobalModelVersion, ...(hospitalId ? {} : { participantHospitalIds: round.participantHospitalIds, submittedHospitalIds: round.submittedHospitalIds }), expectedParticipantCount: round.expectedParticipantCount, submittedParticipantCount: round.submittedParticipantCount, remainingParticipantCount: Math.max(0, round.expectedParticipantCount - round.submittedParticipantCount), totalSamples: round.totalSamples, resultingGlobalModelVersion: round.resultingGlobalModelVersion, createdAt: round.createdAt, startedAt: round.startedAt, readyAt: round.readyAt, aggregationStartedAt: round.aggregationStartedAt, completedAt: round.completedAt, aggregationError: round.aggregationError, isParticipant: hospitalId ? round.participantHospitalIds.includes(hospitalId) : undefined, hasSubmitted: hospitalId ? round.submittedHospitalIds.includes(hospitalId) : undefined });
const participantViews = async (round: any) => {
  if (!round) return [];
  const ids: string[] = round.participantHospitalIds ?? [];
  const users: Array<{ email: string; hospitalName?: string }> = useDb()
    ? (await User.find({ role: 'hospital', email: { $in: ids } }).select('email hospitalName').lean() as unknown as Array<{ email: string; hospitalName?: string }>)
    : ids.map(email => memoryUsers.get(email)).filter((user): user is { email: string; hospitalName?: string; role: Role } => Boolean(user));
  const names = new Map(users.map(user => [user.email, user.hospitalName?.trim() || user.email]));
  return ids.map(hospitalId => ({ hospitalId, hospitalName: names.get(hospitalId) ?? hospitalId, status: round.submittedHospitalIds?.includes(hospitalId) ? 'Submitted' : 'Waiting' }));
};

async function aggregateRound(roundNumber: number, task: MedicalTaskId = 'PNEUMONIA') {
  const claimed: any = useDb() ? await FederatedRound.findOneAndUpdate({ task, round: roundNumber, status: 'READY_FOR_AGGREGATION' }, { $set: { status: 'AGGREGATING', aggregationStartedAt: new Date(), aggregationError: undefined } }, { new: true }).lean() : (() => { const item = memoryRounds.find(round => round.task === task && round.round === roundNumber && round.status === 'READY_FOR_AGGREGATION'); if (item) { item.status = 'AGGREGATING'; item.aggregationStartedAt = new Date(); item.aggregationError = undefined; } return item; })();
  if (!claimed) return;
  await recordAudit('system', 'admin', 'FEDERATED_AGGREGATION_STARTED', `round:${roundNumber}`, { task });
  try {
    const updates: any[] = useDb() ? await ModelUpdate.find({ task, round: roundNumber }).lean() : memoryUpdates.filter(item => item.task === task && item.round === roundNumber);
    if (updates.length !== claimed.expectedParticipantCount || updates.some(update => !claimed.participantHospitalIds.includes(update.hospitalId))) throw new Error('Round does not contain every required participant update.');
    const latest = await currentModel(task);
    if ((latest?.version ?? 0) !== claimed.baseGlobalModelVersion || updates.some(update => update.baseVersion !== claimed.baseGlobalModelVersion)) throw new Error('One or more updates were trained from an outdated global model.');
    const resultingVersion = claimed.baseGlobalModelVersion + 1;
    const aggregate = aggregateWireModels(await Promise.all(updates.map(async update => ({ ...update, binaryPayload: await payloadFor(update) }))), resultingVersion);
    const result: any = { task, version: resultingVersion, round: roundNumber, samples: aggregate.totalSamples, hospitals: updates.length, classes: aggregate.classes, architecture: aggregate.architecture, parameterShapes: aggregate.parameterShapes, status: 'ACTIVE', createdAt: new Date(), checksum: checksum(aggregate.binaryPayload) };
    if (useDb()) {
      const artifactKey = artifactKeyFor('models', task); await saveArtifact(artifactKey, aggregate.binaryPayload); result.artifactKey = artifactKey;
      try {
        await GlobalModel.create(result);
        await GlobalModel.updateMany({ task, status: 'ACTIVE', version: { $ne: result.version } }, { $set: { status: 'SUPERSEDED' } });
        await FederatedRound.updateOne({ _id: claimed._id, status: 'AGGREGATING' }, { $set: { status: 'COMPLETED', resultingGlobalModelVersion: result.version, completedAt: new Date(), totalSamples: result.samples } });
      }
      catch (error) { await removeArtifact(artifactKey); throw error; }
    }
    else { result.binaryPayload = aggregate.binaryPayload; memoryBinaryModels.push(result); const item = memoryRounds.find(round => round.task === task && round.round === roundNumber)!; item.status = 'COMPLETED'; item.resultingGlobalModelVersion = result.version; item.completedAt = new Date(); item.totalSamples = result.samples; }
    await recordAudit('system', 'admin', 'FEDERATED_AGGREGATION_COMPLETED', `round:${roundNumber}`, { task, version: result.version, hospitals: updates.length, samples: result.samples });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Aggregation failed.';
    if (useDb()) await FederatedRound.updateOne({ _id: claimed._id, status: 'AGGREGATING' }, { $set: { status: 'READY_FOR_AGGREGATION', aggregationError: message } }); else { const item = memoryRounds.find(round => round.task === task && round.round === roundNumber)!; item.status = 'READY_FOR_AGGREGATION'; item.aggregationError = message; }
    await recordAudit('system', 'admin', 'FEDERATED_AGGREGATION_FAILED', `round:${roundNumber}`, { task, message });
  }
}
app.post('/api/federated/rounds', auth('admin'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ task: z.string().optional().default('PNEUMONIA'), participantHospitalIds: z.array(z.string().email()).min(1).max(500) }).strict().safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ message: 'Select at least one valid registered hospital.' });
  const task = enabledTaskFor(parsed.data.task); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' });
  if (await activeRound(task.taskId)) return res.status(409).json({ message: `Close or complete the current ${task.displayName} round before creating another.` });
  const participants = [...new Set(parsed.data.participantHospitalIds.map(item => item.toLowerCase()))];
  const registered = useDb() ? (await User.find({ email: { $in: participants }, role: 'hospital' }).select('email').lean()).map(item => item.email) : participants.filter(item => memoryUsers.get(item)?.role === 'hospital');
  if (registered.length !== participants.length) return res.status(400).json({ message: 'Every selected participant must be a registered hospital.' });
  const previous = await currentRound(task.taskId); const model = await currentModel(task.taskId); const now = new Date(); const round: MemoryRound = { _id: new mongoose.Types.ObjectId().toString(), task: task.taskId, round: (previous?.round ?? 0) + 1, baseGlobalModelVersion: model?.version ?? 0, participantHospitalIds: participants, submittedHospitalIds: [], expectedParticipantCount: participants.length, submittedParticipantCount: 0, totalSamples: 0, status: 'ACTIVE', createdAt: now, startedAt: now };
  const saved: any = useDb() ? await FederatedRound.create(round) : (memoryRounds.push(round), round);
  await recordAudit(req.user!.sub, 'admin', 'ROUND_CREATED', `round:${round.round}`, { task: task.taskId, participants: participants.length, baseGlobalModelVersion: round.baseGlobalModelVersion });
  res.status(201).json(roundView(saved));
});
// Version zero is a real AVM1 artifact, not a display-only fallback.  The
// first selected hospital materializes the deterministic local bootstrap once
// it has validated its local class folders; the coordinator then makes that
// exact immutable artifact available to every participant in the round.
app.post('/api/federated/rounds/:round/bootstrap', rateLimit('model-bootstrap', 30), express.raw({ type: ['application/vnd.arogyavaani.model-v1', 'application/octet-stream'], limit: process.env.MODEL_UPDATE_MAX_BYTES ?? '200mb' }), auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ round: z.coerce.number().int().positive(), task: z.string().optional().default('PNEUMONIA') }).safeParse({ round: req.params.round, task: req.query.task });
  const task = parsed.success && enabledTaskFor(parsed.data.task);
  if (!task || !Buffer.isBuffer(req.body) || !req.body.length) return res.status(400).json({ message: 'A task-scoped AVM1 bootstrap payload is required.' });
  const round = await findRound(parsed.data.round, task.taskId);
  if (!round) return res.status(404).json({ message: 'Federated round not found.' });
  if (!['ACTIVE', 'WAITING_FOR_UPDATES'].includes(round.status) || round.baseGlobalModelVersion !== 0) return res.status(409).json({ message: 'This round does not require a version-zero bootstrap model.' });
  if (!round.participantHospitalIds.includes(req.user!.sub)) return res.status(403).json({ message: 'This hospital is not selected for the round.' });
  try {
    const wire = parseWireModel(req.body); validateWireValues(wire);
    const parameterShapes = Object.fromEntries(wire.manifest.tensors.map(tensor => [tensor.name, tensor.shape]));
    if (wire.manifest.task !== task.taskId || wire.manifest.architecture !== task.architecture || wire.manifest.baseVersion !== 0) return res.status(409).json({ message: 'The bootstrap model does not match this task or version.' });
    const existing: any = useDb() ? await GlobalModel.findOne({ task: task.taskId, version: 0 }).lean() : memoryBinaryModels.find(item => item.task === task.taskId && item.version === 0);
    if (existing) {
      const existingClasses = (existing.classes ?? []).map((c: string) => c.toUpperCase()).sort().join('|');
      const incomingClasses = (wire.manifest.classes ?? []).map((c: string) => c.toUpperCase()).sort().join('|');
      const canonicalClasses = (task.classLabels ?? []).map((c: string) => c.toUpperCase()).sort().join('|');

      const existingIsCorrupt = canonicalClasses.length > 0 && existingClasses !== canonicalClasses;
      if (existingIsCorrupt) {
        if (useDb()) {
          await GlobalModel.deleteOne({ _id: existing._id });
          await removeArtifact(existing.artifactKey);
        } else {
          memoryBinaryModels = memoryBinaryModels.filter(item => !(item.task === task.taskId && item.version === 0));
        }
      } else {
        const shapesMatch = sameParameterShapes(existing.parameterShapes, parameterShapes);
        const checksumsMatch = Boolean(existing.checksum && wire.payload && checksum(wire.payload) === existing.checksum);
        if (existingClasses !== incomingClasses || (!shapesMatch && !checksumsMatch)) {
          return res.status(409).json({ message: 'The shared bootstrap model uses different class labels or tensor shapes.' });
        } else {
          return res.status(200).json({ available: true, task: task.taskId, version: 0, round: existing.round, message: 'The shared bootstrap model is already available.' });
        }
      }
    }
    const result: any = { task: task.taskId, version: 0, round: round.round, samples: 0, hospitals: 0, classes: wire.manifest.classes, architecture: wire.manifest.architecture, parameterShapes, status: 'BOOTSTRAP', createdAt: new Date(), checksum: checksum(req.body) };
    if (useDb()) {
      const artifactKey = artifactKeyFor('models', task.taskId); await saveArtifact(artifactKey, req.body);
      try { await GlobalModel.create({ ...result, artifactKey }); } catch (error) { await removeArtifact(artifactKey); if (await GlobalModel.exists({ task: task.taskId, version: 0 })) return res.status(200).json({ available: true, task: task.taskId, version: 0, message: 'The shared bootstrap model is already available.' }); throw error; }
    } else {
      if (memoryBinaryModels.some(item => item.task === task.taskId && item.version === 0)) return res.status(200).json({ available: true, task: task.taskId, version: 0, message: 'The shared bootstrap model is already available.' });
      result.binaryPayload = req.body; memoryBinaryModels.push(result);
    }
    await recordAudit(req.user!.sub, 'hospital', 'BOOTSTRAP_MODEL_PUBLISHED', `round:${round.round}`, { task: task.taskId, payloadBytes: req.body.length });
    res.status(201).json({ available: true, task: task.taskId, version: 0, round: round.round, bytes: req.body.length });
  } catch (error) { res.status(400).json({ message: error instanceof Error ? error.message : 'Invalid bootstrap model payload.' }); }
});
app.get('/api/federated/rounds', auth('admin'), async (req, res) => { const task = req.query.task ? enabledTaskFor(req.query.task) : undefined; if (req.query.task && !task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); const filter = task ? { task: task.taskId } : {}; res.json(((useDb() ? await FederatedRound.find(filter).sort({ round: -1 }).lean() : [...memoryRounds].filter(round => !task || round.task === task.taskId).reverse()) as any[]).map(round => roundView(round))); });
app.get('/api/federated/rounds/active', auth(), async (req: AuthenticatedRequest, res) => {
  const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' });
  const round = await activeRound(task.taskId) ?? (req.user!.role === 'hospital' ? (useDb() ? await FederatedRound.findOne({ task: task.taskId, participantHospitalIds: req.user!.sub }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => item.task === task.taskId && item.participantHospitalIds.includes(req.user!.sub))) : undefined);
  if (!round) return res.status(404).json({ message: 'No active federated round is available.' });
  res.json(roundView(round, req.user!.role === 'hospital' ? req.user!.sub : undefined));
});
app.get('/api/federated/rounds/:round', auth(), async (req: AuthenticatedRequest, res) => { const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); const round = await findRound(Number(req.params.round), task.taskId); if (!round) return res.status(404).json({ message: 'Federated round not found.' }); if (req.user!.role === 'hospital' && !round.participantHospitalIds.includes(req.user!.sub)) return res.status(403).json({ message: 'You are not a participant in this round.' }); res.json(roundView(round, req.user!.role === 'hospital' ? req.user!.sub : undefined)); });
app.post('/api/federated/rounds/:round/cancel', auth('admin'), async (req: AuthenticatedRequest, res) => { const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); const round = await findRound(Number(req.params.round), task.taskId); if (!round) return res.status(404).json({ message: 'Federated round not found.' }); if (!['DRAFT', 'ACTIVE', 'WAITING_FOR_UPDATES', 'READY_FOR_AGGREGATION'].includes(round.status)) return res.status(409).json({ message: 'Only an incomplete round can be cancelled.' }); if (useDb()) await FederatedRound.updateOne({ _id: round._id, status: round.status }, { $set: { status: 'CANCELLED', completedAt: new Date() } }); else { round.status = 'CANCELLED'; round.completedAt = new Date(); } await recordAudit(req.user!.sub, 'admin', 'ROUND_CANCELLED', `round:${round.round}`, { task: task.taskId }); res.json(roundView(await findRound(round.round, task.taskId))); });
async function submitRoundUpdate(req: AuthenticatedRequest, res: express.Response, roundNumber: number, samples: number, taskId: MedicalTaskId = 'PNEUMONIA') {
  if (!Buffer.isBuffer(req.body) || !req.body.length) return res.status(400).json({ message: 'An AVM1 model payload is required.' });
  const task = enabledTaskFor(taskId); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' });
  const round = await findRound(roundNumber, task.taskId); if (!round) return res.status(404).json({ message: 'Federated round not found.' });
  if (!['ACTIVE', 'WAITING_FOR_UPDATES'].includes(round.status)) return res.status(409).json({ message: `Round is ${round.status} and cannot accept updates.` });
  if (!round.participantHospitalIds.includes(req.user!.sub)) return res.status(403).json({ message: 'This hospital is not selected for the round.' });
  if (round.submittedHospitalIds.includes(req.user!.sub)) return res.status(409).json({ message: 'This hospital has already submitted an update for this round.' });
  try {
    const wire = parseWireModel(req.body); validateWireValues(wire); const parameterShapes = Object.fromEntries(wire.manifest.tensors.map(tensor => [tensor.name, tensor.shape]));
    // AVM1 task identity is required for every new task-aware submission. The
    // original pneumonia transport can remain compatible because it is the
    // only pre-existing, unambiguous task.
    if (!wire.manifest.task && task.taskId !== 'PNEUMONIA') return res.status(409).json({ message: 'The submitted model is missing its required medical task identity.' });
    if (wire.manifest.task && wire.manifest.task !== task.taskId) return res.status(409).json({ message: 'The submitted model belongs to a different medical task.' });
    if (wire.manifest.architecture !== task.architecture) return res.status(409).json({ message: 'The submitted model architecture does not match this medical task.' });
    if (wire.manifest.baseVersion !== round.baseGlobalModelVersion) return res.status(409).json({ message: `This round requires global model v${round.baseGlobalModelVersion}. Synchronize and train again.` });
    const model = await currentModel(task.taskId); if (model && (model.version !== round.baseGlobalModelVersion || model.architecture !== wire.manifest.architecture || model.classes?.join('|') !== wire.manifest.classes.join('|') || !sameParameterShapes(model.parameterShapes, parameterShapes))) return res.status(409).json({ message: 'The update is incompatible with the required global model.' });
    // Before a task has a global model, the first accepted update establishes
    // a round-local signature. Reject incompatible peer updates immediately.
    const priorUpdate: any = useDb()
      ? await ModelUpdate.findOne({ task: task.taskId, round: roundNumber }).lean()
      : memoryUpdates.find(item => item.task === task.taskId && item.round === roundNumber);
    if (priorUpdate && (priorUpdate.architecture !== wire.manifest.architecture || priorUpdate.classes.join('|') !== wire.manifest.classes.join('|') || !sameParameterShapes(priorUpdate.parameterShapes, parameterShapes))) {
      return res.status(409).json({ message: 'The update is incompatible with model classes or tensor shapes already submitted for this task round.' });
    }
    const update: MemoryUpdate = { task: task.taskId, hospitalId: req.user!.sub, round: roundNumber, roundId: String(round._id), samples, binaryPayload: req.body, classes: wire.manifest.classes, architecture: wire.manifest.architecture, baseVersion: wire.manifest.baseVersion, parameterShapes, submittedAt: new Date() };
    let storedArtifactKey: string | undefined;
    try {
      if (useDb()) {
        storedArtifactKey = artifactKeyFor('updates', task.taskId); await saveArtifact(storedArtifactKey, req.body);
        await ModelUpdate.create({ ...update, binaryPayload: undefined, artifactKey: storedArtifactKey, checksum: checksum(req.body), payloadBytes: req.body.length });
      } else { if (memoryUpdates.some(item => item.task === task.taskId && item.round === roundNumber && item.hospitalId === update.hospitalId)) throw new Error('duplicate'); memoryUpdates.push(update); }
    } catch (err: any) {
      await removeArtifact(storedArtifactKey);
      if (err?.code === 11000 || err?.message === 'duplicate') return res.status(409).json({ message: 'This hospital has already submitted an update for this round.' });
      console.warn('Failed to persist model update:', err?.message ?? err);
      return res.status(500).json({ message: 'Failed to persist model update artifact: ' + (err instanceof Error ? err.message : 'Storage error') });
    }
    let updated: any;
    if (useDb()) updated = await FederatedRound.findOneAndUpdate({ _id: round._id, status: { $in: ['ACTIVE', 'WAITING_FOR_UPDATES'] }, submittedHospitalIds: { $ne: req.user!.sub } }, { $addToSet: { submittedHospitalIds: req.user!.sub }, $inc: { submittedParticipantCount: 1, totalSamples: samples }, $set: { status: 'WAITING_FOR_UPDATES' } }, { new: true }).lean(); else { const item = memoryRounds.find(value => value.task === task.taskId && value.round === roundNumber)!; if (!item.submittedHospitalIds.includes(req.user!.sub)) { item.submittedHospitalIds.push(req.user!.sub); item.submittedParticipantCount++; item.totalSamples += samples; item.status = 'WAITING_FOR_UPDATES'; updated = item; } }
    if (!updated) { if (useDb()) { await ModelUpdate.deleteOne({ task: task.taskId, round: roundNumber, hospitalId: req.user!.sub }); await removeArtifact(storedArtifactKey); } else memoryUpdates = memoryUpdates.filter(item => item !== update); return res.status(409).json({ message: 'Round state changed; update was not accepted.' }); }
    if (updated.submittedParticipantCount === updated.expectedParticipantCount) { if (useDb()) updated = await FederatedRound.findOneAndUpdate({ _id: updated._id, status: 'WAITING_FOR_UPDATES' }, { $set: { status: 'READY_FOR_AGGREGATION', readyAt: new Date() } }, { new: true }).lean() ?? updated; else { updated.status = 'READY_FOR_AGGREGATION'; updated.readyAt = new Date(); } void aggregateRound(roundNumber, task.taskId); }
    await recordAudit(req.user!.sub, 'hospital', 'MODEL_UPDATE_SUBMITTED', `round:${roundNumber}`, { task: task.taskId, samples, payloadBytes: req.body.length });
    res.status(202).json({ accepted: true, bytes: req.body.length, message: updated.status === 'READY_FOR_AGGREGATION' ? 'Final required update received; aggregation has started.' : 'Model update stored. Waiting for other participating hospitals.', ...roundView(updated, req.user!.sub) });
  } catch (error) { await recordAudit(req.user!.sub, 'hospital', 'federated.invalid_update_rejected', `round:${roundNumber}`); res.status(400).json({ message: error instanceof Error ? error.message : 'Invalid binary model payload.' }); }
}
app.post('/api/federated/rounds/:round/updates', rateLimit('model-update', 120), express.raw({ type: ['application/vnd.arogyavaani.model-v1', 'application/octet-stream'], limit: process.env.MODEL_UPDATE_MAX_BYTES ?? '200mb' }), auth('hospital'), async (req: AuthenticatedRequest, res) => { const parsed = z.object({ round: z.coerce.number().int().positive(), samples: z.coerce.number().int().positive(), task: z.string().optional().default('PNEUMONIA') }).safeParse({ round: req.params.round, samples: req.query.samples, task: req.query.task }); if (!parsed.success || !enabledTaskFor(parsed.data.task)) return res.status(400).json({ message: 'A positive sample count and enabled medical task are required.' }); await submitRoundUpdate(req, res, parsed.data.round, parsed.data.samples, parsed.data.task as MedicalTaskId); });
app.post('/api/federated/update-binary', express.raw({ type: ['application/vnd.arogyavaani.model-v1', 'application/octet-stream'], limit: process.env.MODEL_UPDATE_MAX_BYTES ?? '200mb' }), auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ samples: z.coerce.number().int().positive(), round: z.coerce.number().int().positive(), task: z.string().optional().default('PNEUMONIA') }).safeParse(req.query);
  if (!parsed.success || !enabledTaskFor(parsed.data.task)) return res.status(400).json({ message: 'A positive round, sample count, and enabled medical task are required.' });
  await submitRoundUpdate(req, res, parsed.data.round, parsed.data.samples, parsed.data.task as MedicalTaskId);
});
app.post('/api/federated/update', auth('hospital'), async (req: AuthenticatedRequest, res) => {
  res.status(410).json({ message: 'JSON parameter updates are retired. Submit an AVM1 payload to the selected federated round.' });
});
app.get('/api/federated/status', auth(), async (req: AuthenticatedRequest, res) => {
  const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' });
  const model = await currentModel(task.taskId); const round: any = await activeRound(task.taskId) ?? (req.user!.role === 'hospital' ? (useDb() ? await FederatedRound.findOne({ task: task.taskId, participantHospitalIds: req.user!.sub }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => item.task === task.taskId && item.participantHospitalIds.includes(req.user!.sub))) : undefined);
  res.json({ task: task.taskId, version: model?.version ?? 0, modelStatus: model?.status ?? 'No model available', architecture: model?.architecture ?? task.architecture, round: round?.round ?? 0, samples: model?.samples ?? 0, hospitals: round?.expectedParticipantCount ?? 0, updatesReceived: round?.submittedParticipantCount ?? 0, aggregationStatus: round?.status ?? 'NO_ACTIVE_ROUND', updatedAt: model?.updatedAt ?? model?.createdAt ?? new Date(), baseGlobalModelVersion: round?.baseGlobalModelVersion, isParticipant: round && req.user!.role === 'hospital' ? round.participantHospitalIds.includes(req.user!.sub) : undefined, hasSubmitted: round && req.user!.role === 'hospital' ? round.submittedHospitalIds.includes(req.user!.sub) : undefined, ...(req.user!.role === 'admin' ? { participants: await participantViews(round) } : {}) });
});
app.post('/api/federated/aggregate', auth('admin'), async (_req, res) => res.status(410).json({ message: 'Aggregation is automatic and waits for every selected participant.' }));
app.post('/api/federated/aggregate-binary', auth('admin'), async (_req, res) => res.status(410).json({ message: 'Aggregation is automatic and waits for every selected participant.' }));
app.post('/api/federated/rounds/:round/aggregate', auth('admin'), async (req, res) => { const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); const round = await findRound(Number(req.params.round), task.taskId); if (!round) return res.status(404).json({ message: 'Federated round not found.' }); if (round.status !== 'READY_FOR_AGGREGATION') return res.status(409).json({ message: 'Aggregation requires every selected participant update.' }); void aggregateRound(round.round, task.taskId); res.status(202).json({ message: 'Aggregation is running.' }); });
const hospitalMayAccessModel = async (hospitalId: string, model: any, task: MedicalTaskId) => {
  const originatingRound = await findRound(model.round, task);
  if (originatingRound?.participantHospitalIds?.includes(hospitalId)) return true;
  // A newly selected participant must be able to fetch the exact immutable
  // base model for its assigned round, even if it did not train the round that
  // originally produced that model.  Do not grant access merely because a
  // hospital has an account or participated in another task/version.
  return useDb()
    ? Boolean(await FederatedRound.exists({ task, baseGlobalModelVersion: model.version, participantHospitalIds: hospitalId, status: { $in: openStatuses } }))
    : memoryRounds.some(round => round.task === task && round.baseGlobalModelVersion === model.version && round.participantHospitalIds.includes(hospitalId) && openStatuses.includes(round.status));
};
app.get('/api/federated/global-model', auth('hospital'), async (req: AuthenticatedRequest, res) => { const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); const model = await currentModel(task.taskId); if (!model) return res.status(404).json({ message: `No ${task.displayName} global model has been aggregated yet.` }); if (!await hospitalMayAccessModel(req.user!.sub, model, task.taskId)) return res.status(403).json({ message: 'This hospital is not authorized to access this task model.' }); res.json(modelMetadata(model)); });
app.get('/api/federated/global-model-binary', auth('hospital'), async (req, res) => {
  const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' });
  const version = z.coerce.number().int().nonnegative().safeParse(req.query.version);
  const model: any = version.success
    ? (useDb() ? await GlobalModel.findOne({ task: task.taskId, version: version.data }).lean() : memoryBinaryModels.find(item => item.task === task.taskId && item.version === version.data))
    : await currentModel(task.taskId);
  if (!model) return res.status(404).json({ message: 'No binary global model has been aggregated yet.' });
  if (!await hospitalMayAccessModel((req as AuthenticatedRequest).user!.sub, model, task.taskId)) return res.status(403).json({ message: 'This hospital is not authorized to access this task model.' });
  try {
    const payload = await payloadFor(model);
    res.setHeader('Content-Type', 'application/vnd.arogyavaani.model-v1'); res.setHeader('Content-Length', payload.length); res.setHeader('X-ArogyaVaani-Model-Version', String(model.version)); res.setHeader('X-ArogyaVaani-Task', task.taskId); if (model.checksum) res.setHeader('X-ArogyaVaani-Model-SHA256', model.checksum); res.send(payload);
  } catch { res.status(503).json({ message: 'The global model artifact is temporarily unavailable.' }); }
});
app.post('/api/training/runs', auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ task: z.string().optional().default('PNEUMONIA'), status: z.enum(['queued', 'validating', 'training', 'completed', 'failed']), samples: z.number().int().nonnegative().optional(), epochs: z.number().int().positive().optional(), loss: z.number().finite().optional(), accuracy: z.number().finite().optional(), checkpoint: z.string().max(512).optional(), message: z.string().max(1000).optional(), startedAt: z.string().datetime().optional(), completedAt: z.string().datetime().optional() }).safeParse(req.body);
  const task = parsed.success && enabledTaskFor(parsed.data.task); if (!parsed.success || !task) return res.status(400).json({ message: 'Invalid task or training run metadata.' });
  const run = { ...parsed.data, task: task.taskId, hospitalId: req.user!.sub, startedAt: parsed.data.startedAt ? new Date(parsed.data.startedAt) : undefined, completedAt: parsed.data.completedAt ? new Date(parsed.data.completedAt) : undefined };
  if (useDb()) await TrainingRun.create(run); else memoryTrainingRuns.push({ ...run, createdAt: new Date() });
  await recordAudit(req.user!.sub, 'hospital', 'training.run_recorded', undefined, { task: task.taskId, status: run.status, samples: run.samples });
  res.status(201).json({ recorded: true });
});
app.get('/api/federated/history', auth(), async (req, res) => { const task = req.query.task ? enabledTaskFor(req.query.task) : undefined; if (req.query.task && !task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); res.json(useDb() ? await FederatedRound.find(task ? { task: task.taskId } : {}).sort({ round: -1 }).lean() : [...memoryRounds].filter(round => !task || round.task === task.taskId).reverse()); });
app.get('/api/training/runs', auth('hospital'), async (req: AuthenticatedRequest, res) => { const task = req.query.task ? enabledTaskFor(req.query.task) : undefined; if (req.query.task && !task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); const filter = { hospitalId: req.user!.sub, ...(task ? { task: task.taskId } : {}) }; res.json(useDb() ? await TrainingRun.find(filter).sort({ createdAt: -1 }).lean() : memoryTrainingRuns.filter(run => run.hospitalId === req.user!.sub && (!task || run.task === task.taskId)).reverse()); });
// The local service remains hospital-controlled and loopback-only.  It proves
// reachability by having the authenticated hospital browser send this short
// heartbeat after calling its local health endpoint; the coordinator never
// attempts an unsafe inbound connection to a hospital network.
app.post('/api/hospital/agent/heartbeat', auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const parsed = z.object({ tasks: z.array(z.string()).max(20).optional().default([]) }).strict().safeParse(req.body);
  if (!parsed.success || parsed.data.tasks.some(taskId => !enabledTaskFor(taskId))) return res.status(400).json({ message: 'Connected-agent tasks are invalid.' });
  const now = new Date();
  if (useDb()) await User.updateOne({ email: req.user!.sub, role: 'hospital' }, { $set: { agentLastSeen: now, agentTasks: parsed.data.tasks } });
  else { const user = memoryUsers.get(req.user!.sub); if (user) { user.agentLastSeen = now; user.agentTasks = parsed.data.tasks; } }
  res.json({ connected: true, lastSeen: now, tasks: parsed.data.tasks });
});
app.get('/api/hospital/overview', auth('hospital'), async (req: AuthenticatedRequest, res) => {
  const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' });
  const user: any = useDb() ? await User.findOne({ email: req.user!.sub }).lean() : memoryUsers.get(req.user!.sub);
  const model = await currentModel(task.taskId); const round: any = await activeRound(task.taskId) ?? (useDb() ? await FederatedRound.findOne({ task: task.taskId, participantHospitalIds: req.user!.sub }).sort({ round: -1 }).lean() : [...memoryRounds].reverse().find(item => item.task === task.taskId && item.participantHospitalIds.includes(req.user!.sub)));
  const lastUpdate: any = useDb() ? await ModelUpdate.findOne({ task: task.taskId, hospitalId: req.user!.sub }).sort({ submittedAt: -1 }).lean() : memoryUpdates.filter(update => update.task === task.taskId && update.hospitalId === req.user!.sub).sort((a, b) => b.submittedAt.getTime() - a.submittedAt.getTime())[0];
  const lastTraining: any = useDb() ? await TrainingRun.findOne({ task: task.taskId, hospitalId: req.user!.sub }).sort({ createdAt: -1 }).lean() : memoryTrainingRuns.filter(run => run.task === task.taskId && run.hospitalId === req.user!.sub).at(-1);
  const bootstrapModel = !model && round?.participantHospitalIds.includes(req.user!.sub) && round.baseGlobalModelVersion === 0
    ? { task: task.taskId, version: 0, round: round.round, status: 'BOOTSTRAP', updatedAt: round.startedAt ?? round.createdAt }
    : null;
  const agentOnline = Boolean(user?.agentLastSeen && Date.now() - new Date(user.agentLastSeen).getTime() < 30_000);
  res.json({ task: task.taskId, hospitalName: user?.hospitalName ?? 'Hospital', connectionStatus: databaseState === 'error' ? 'Degraded' : agentOnline ? 'Connected' : 'Local agent offline', globalModel: model ? { task: task.taskId, version: model.version, round: model.round, status: model.status, updatedAt: model.updatedAt ?? model.createdAt } : bootstrapModel, currentRound: round?.round ?? 0, round: roundView(round, req.user!.sub), training: lastTraining ? { status: lastTraining.status, samples: lastTraining.samples, loss: lastTraining.loss, accuracy: lastTraining.accuracy, updatedAt: lastTraining.completedAt ?? lastTraining.createdAt } : { status: 'No local training recorded' }, participationStatus: round ? (round.submittedHospitalIds.includes(req.user!.sub) ? 'Update submitted for this round' : round.participantHospitalIds.includes(req.user!.sub) ? 'Awaiting local update' : 'Not selected for this round') : `No active ${task.displayName} round`, lastContribution: lastUpdate ? { round: lastUpdate.round, samples: lastUpdate.samples, receivedAt: lastUpdate.submittedAt } : null });
});
app.get('/api/admin/dashboard', auth('admin'), async (req, res) => { const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); if (!task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' }); const hospitals = useDb() ? await User.countDocuments({ role: 'hospital' }) : [...memoryUsers.values()].filter(user => user.role === 'hospital').length; const model = await currentModel(task.taskId); const round: any = await activeRound(task.taskId); res.json({ task: task.taskId, totalHospitals: hospitals, activeHospitals: round?.submittedParticipantCount ?? 0, currentRound: round?.round ?? 0, globalVersion: model?.version ?? 0, updatesReceived: round?.submittedParticipantCount ?? 0, activeRound: roundView(round), systemHealth: 'Operational' }); });
app.get('/api/admin/hospitals', auth('admin'), async (_req, res) => res.json(useDb() ? await User.find({ role: 'hospital' }).select('email hospitalName createdAt').lean() : [...memoryUsers.values()].filter(user => user.role === 'hospital').map(({ passwordHash, ...user }) => user)));
app.get('/api/admin/federated-rounds', auth('admin'), async (req, res) => { const task = req.query.task ? enabledTaskFor(req.query.task) : undefined; res.json(useDb() ? await FederatedRound.find(task ? { task: task.taskId } : {}).sort({ round: -1 }).lean() : [...memoryRounds].filter(round => !task || round.task === task.taskId).reverse()); });
app.get('/api/admin/models', auth('admin'), async (req, res) => { const task = req.query.task ? enabledTaskFor(req.query.task) : undefined; res.json((useDb() ? await GlobalModel.find(task ? { task: task.taskId } : {}).sort({ version: -1 }).lean() : [...memoryModels, ...memoryBinaryModels].filter(model => !task || model.task === task.taskId).sort((left, right) => right.version - left.version)).map(modelMetadata)); });
app.post('/api/admin/models/:version/rollback', auth('admin'), async (req: AuthenticatedRequest, res) => {
  const task = enabledTaskFor(req.query.task ?? 'PNEUMONIA'); const version = z.coerce.number().int().positive().safeParse(req.params.version);
  if (!task || !version.success) return res.status(400).json({ message: 'A valid medical task and model version are required.' });
  const model: any = useDb() ? await GlobalModel.findOne({ task: task.taskId, version: version.data }).lean() : [...memoryModels, ...memoryBinaryModels].find(item => item.task === task.taskId && item.version === version.data);
  if (!model) return res.status(404).json({ message: 'The requested task-specific model version does not exist.' });
  if (useDb()) {
    await GlobalModel.updateMany({ task: task.taskId, status: 'ACTIVE', version: { $ne: version.data } }, { $set: { status: 'SUPERSEDED' } });
    await GlobalModel.updateOne({ _id: model._id }, { $set: { status: 'ACTIVE' } });
  } else {
    for (const item of [...memoryModels, ...memoryBinaryModels]) if (item.task === task.taskId && item.status === 'ACTIVE') item.status = 'SUPERSEDED';
    model.status = 'ACTIVE';
  }
  await recordAudit(req.user!.sub, 'admin', 'GLOBAL_MODEL_ROLLED_BACK', `model:${version.data}`, { task: task.taskId, version: version.data });
  res.json({ rolledBack: true, model: modelMetadata(useDb() ? await GlobalModel.findOne({ task: task.taskId, version: version.data }).lean() : model) });
});
app.get('/api/admin/model-updates', auth('admin'), async (req, res) => { const task = req.query.task ? enabledTaskFor(req.query.task) : undefined; res.json(((useDb() ? await ModelUpdate.find(task ? { task: task.taskId } : {}).sort({ submittedAt: -1 }).lean() : [...memoryUpdates].filter(update => !task || update.task === task.taskId).sort((left, right) => right.submittedAt.getTime() - left.submittedAt.getTime())) as any[]).map(updateMetadata)); });
app.get('/api/admin/audit-logs', auth('admin'), async (req, res) => {
  const task = req.query.task ? enabledTaskFor(req.query.task) : undefined;
  if (req.query.task && !task) return res.status(400).json({ message: 'Medical task is invalid or disabled.' });
  const filter = task ? { 'details.task': task.taskId } : {};
  res.json(useDb() ? await AuditLog.find(filter).sort({ createdAt: -1 }).limit(100).lean() : [...memoryAuditLogs].filter(item => !task || (item.details as Record<string, unknown> | undefined)?.task === task.taskId).reverse().slice(0, 100));
});
app.post('/api/predictions/metadata', rateLimit('prediction-metadata', 120), auth('hospital'), async (req: AuthenticatedRequest, res) => { const parsed = z.object({ task: z.string().optional().default('PNEUMONIA'), modelVersion: z.number().int().nonnegative(), predictedClass: z.string().min(1), confidence: z.number().min(0).max(1) }).safeParse(req.body); const task = parsed.success && enabledTaskFor(parsed.data.task); if (!parsed.success || !task) return res.status(400).json({ message: 'Only task-specific prediction metadata may be recorded.' }); if (useDb()) await Prediction.create({ ...parsed.data, task: task.taskId, hospitalId: req.user!.sub }); await recordAudit(req.user!.sub, 'hospital', 'PREDICTION_COMPLETED', undefined, { task: task.taskId, modelVersion: parsed.data.modelVersion }); res.status(201).json({ recorded: true }); });
// Keep parser and implementation errors out of API responses.  Details are
// recorded server-side while callers receive a stable, non-sensitive message.
app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  const status = typeof error === 'object' && error && 'status' in error && typeof (error as { status?: unknown }).status === 'number' ? (error as { status: number }).status : 500;
  console.warn(JSON.stringify({ event: 'API_ERROR', status, message: error instanceof Error ? error.message : 'Unknown error' }));
  res.status(status >= 400 && status < 600 ? status : 500).json({ message: status === 413 ? 'Request body exceeds the configured size limit.' : 'The request could not be completed safely.' });
});
if (process.env.NODE_ENV !== 'test') app.listen(port, '0.0.0.0', () => console.log(`ArogyaVaani API listening on http://0.0.0.0:${port}`));
