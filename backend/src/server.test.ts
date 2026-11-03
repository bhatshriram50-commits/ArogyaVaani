import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import type { Server } from 'node:http';

// Keep tests independent of developer OAuth/Mongo environment files.
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-secret-with-sufficient-length';
process.env.ADMIN_EMAIL = 'admin@example.org';
process.env.ADMIN_PASSWORD = 'Admin-password-123!';
process.env.GOOGLE_CLIENT_ID = '';
process.env.DOTENV_CONFIG_PATH = 'missing-test-env-file';

const { app } = await import('./server.js');
let server: Server; let baseUrl = ''; let adminToken = ''; const tokens = new Map<string, string>();
const hospitals = ['hospital-a@example.org', 'hospital-b@example.org', 'hospital-c@example.org', 'hospital-d@example.org'];
const request = async (path: string, init: RequestInit = {}) => { const response = await fetch(baseUrl + path, init); const isJson = response.headers.get('content-type')?.includes('application/json'); const body = response.status === 204 || !isJson ? undefined : await response.json() as any; return { response, body }; };
const json = (body: object, token?: string): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) }, body: JSON.stringify(body) });
const payload = (value: number, baseVersion: number, architecture = 'densenet121', task?: string) => { const data = Buffer.alloc(4); data.writeFloatLE(value); const manifest = Buffer.from(JSON.stringify({ format: 'arogyavaani-state-v1', ...(task ? { task } : {}), architecture, baseVersion, classes: ['negative', 'positive'], tensors: [{ name: 'weight', shape: [1], dtype: 'f32', offset: 0, length: 4 }] })); const header = Buffer.alloc(8); header.write('AVM1'); header.writeUInt32LE(manifest.length, 4); return Buffer.concat([header, manifest, data]); };
const submit = (round: number, email: string, value: number, samples: number, baseVersion = 0, architecture = 'densenet121') => request(`/api/federated/rounds/${round}/updates?samples=${samples}`, { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(email)}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(value, baseVersion, architecture) });
const publishBootstrap = (round: number, email: string) => request(`/api/federated/rounds/${round}/bootstrap`, { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(email)}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(0, 0, 'densenet121', 'PNEUMONIA') });
const waitForCompletion = async (round: number) => { for (let attempt = 0; attempt < 40; attempt++) { const state = await request(`/api/federated/rounds/${round}`, { headers: { Authorization: `Bearer ${adminToken}` } }); if (state.body.status === 'COMPLETED') return state.body; await new Promise(resolve => setTimeout(resolve, 10)); } throw new Error('round did not complete'); };

before(async () => {
  server = app.listen(0, '127.0.0.1'); await new Promise<void>(resolve => server.once('listening', resolve)); const address = server.address(); assert.ok(address && typeof address !== 'string'); baseUrl = `http://127.0.0.1:${address.port}`;
  for (const email of hospitals) { assert.equal((await request('/api/auth/register', json({ hospitalName: email, email, password: 'Hospital-password-123!' }))).response.status, 201); const login = await request('/api/auth/login', json({ email, password: 'Hospital-password-123!' })); tokens.set(email, login.body.token); }
  const login = await request('/api/auth/login', json({ email: 'admin@example.org', password: 'Admin-password-123!' })); adminToken = login.body.token;
});
after(() => server.close());

test('Google sign-in is deterministically unavailable without a configured verifier', async () => assert.equal((await request('/api/auth/google', json({ idToken: 'unverified-token' }))).response.status, 503));
test('three selected hospitals submit asynchronously and aggregate exactly once', async () => {
  const created = await request('/api/federated/rounds', json({ participantHospitalIds: hospitals.slice(0, 3) }, adminToken)); assert.equal(created.response.status, 201); assert.equal(created.body.baseGlobalModelVersion, 0); assert.equal(created.body.expectedParticipantCount, 3);
  for (const email of hospitals.slice(0, 3)) { const active = await request('/api/federated/rounds/active', { headers: { Authorization: `Bearer ${tokens.get(email)}` } }); assert.equal(active.body.isParticipant, true); assert.equal(active.body.round, 1); }
  assert.equal((await publishBootstrap(1, hospitals[0])).response.status, 201); assert.equal((await request('/api/federated/global-model-binary?version=0', { headers: { Authorization: `Bearer ${tokens.get(hospitals[1])}` } })).response.status, 200);
  const first = await submit(1, hospitals[0], 1, 1); assert.equal(first.response.status, 202); assert.equal(first.body.status, 'WAITING_FOR_UPDATES'); assert.equal(first.body.submittedParticipantCount, 1);
  const second = await submit(1, hospitals[1], 3, 3); assert.equal(second.response.status, 202); assert.equal(second.body.submittedParticipantCount, 2);
  const final = await submit(1, hospitals[2], 5, 2); assert.equal(final.response.status, 202); assert.ok(['READY_FOR_AGGREGATION', 'AGGREGATING'].includes(final.body.status));
  const completed = await waitForCompletion(1); assert.equal(completed.resultingGlobalModelVersion, 1); assert.equal(completed.submittedParticipantCount, 3); assert.equal(completed.totalSamples, 6);
  const model = await request('/api/federated/global-model', { headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}` } }); assert.equal(model.body.version, 1); assert.equal(model.body.hospitals, 3);
});
test('duplicate, nonparticipant, stale, incompatible and cancelled round updates are rejected', async () => {
  const roundTwo = await request('/api/federated/rounds', json({ participantHospitalIds: hospitals.slice(0, 2) }, adminToken)); assert.equal(roundTwo.response.status, 201); assert.equal(roundTwo.body.baseGlobalModelVersion, 1);
  assert.equal((await submit(2, hospitals[0], 2, 1, 0)).response.status, 409); assert.equal((await submit(2, hospitals[0], 2, 1, 1, 'othernet')).response.status, 409); assert.equal((await submit(2, hospitals[3], 2, 1, 1)).response.status, 403);
  assert.equal((await submit(2, hospitals[0], 2, 1, 1)).response.status, 202); assert.equal((await submit(2, hospitals[0], 2, 1, 1)).response.status, 409);
  assert.equal((await request('/api/federated/rounds/2/cancel', json({}, adminToken))).response.status, 200); assert.equal((await submit(2, hospitals[1], 6, 1, 1)).response.status, 409);
});
test('invalid tensor values and malformed payloads are rejected before a round progresses', async () => {
  const created = await request('/api/federated/rounds', json({ participantHospitalIds: [hospitals[0]] }, adminToken)); assert.equal(created.response.status, 201);
  assert.equal((await request('/api/federated/rounds/3/updates?samples=1', { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: Buffer.from('not-an-avm1-model') })).response.status, 400);
  assert.equal((await submit(3, hospitals[0], Number.NaN, 1, 1)).response.status, 400);
  assert.equal((await request('/api/federated/rounds/3', { headers: { Authorization: `Bearer ${adminToken}` } })).body.submittedParticipantCount, 0);
  assert.equal((await request('/api/federated/rounds/3/cancel', json({}, adminToken))).response.status, 200);
});
test('concurrent final submissions claim aggregation once and create one global version', async () => {
  const created = await request('/api/federated/rounds', json({ participantHospitalIds: hospitals.slice(0, 3) }, adminToken)); assert.equal(created.response.status, 201); assert.equal(created.body.round, 4); assert.equal(created.body.baseGlobalModelVersion, 1);
  assert.equal((await submit(4, hospitals[0], 1, 1, 1)).response.status, 202);
  const responses = await Promise.all([submit(4, hospitals[1], 3, 1, 1), submit(4, hospitals[2], 5, 2, 1)]); assert.ok(responses.every(result => result.response.status === 202));
  const completed = await waitForCompletion(4); assert.equal(completed.resultingGlobalModelVersion, 2); assert.equal(completed.submittedParticipantCount, 3);
  const models = await request('/api/admin/models', { headers: { Authorization: `Bearer ${adminToken}` } }); assert.equal(models.body.filter((model: any) => model.version === 2).length, 1);
});
test('raw images and JSON parameter updates are never accepted by the central coordinator', async () => { assert.equal((await request('/api/federated/update', json({ round: 2, samples: 1, image: 'base64-xray' }, tokens.get(hospitals[0])))).response.status, 410); });
test('tasks have independent round numbers, model versions, and reject cross-task AVM1 updates', async () => {
  const retinopathy = await request('/api/federated/rounds', json({ task: 'RETINOPATHY', participantHospitalIds: [hospitals[0]] }, adminToken));
  const brain = await request('/api/federated/rounds', json({ task: 'BRAIN_TUMOR', participantHospitalIds: [hospitals[1]] }, adminToken));
  assert.equal(retinopathy.response.status, 201); assert.equal(brain.response.status, 201);
  assert.equal(retinopathy.body.round, 1); assert.equal(brain.body.round, 1);
  const cross = await request('/api/federated/rounds/1/updates?samples=1&task=RETINOPATHY', { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(1, 0, 'densenet121', 'PNEUMONIA') });
  assert.equal(cross.response.status, 409);
  const missingTask = await request('/api/federated/rounds/1/updates?samples=1&task=RETINOPATHY', { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(1, 0) });
  assert.equal(missingTask.response.status, 409);
  const invalidTask = await request('/api/federated/rounds/1/updates?samples=1&task=RETINOPATHY', { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(1, 0, 'densenet121', 'UNKNOWN') });
  assert.equal(invalidTask.response.status, 400);
  const retSubmission = await request('/api/federated/rounds/1/updates?samples=1&task=RETINOPATHY', { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(2, 0, 'densenet121', 'RETINOPATHY') }); assert.equal(retSubmission.response.status, 202, retSubmission.body.message);
  assert.equal((await request('/api/federated/rounds/1/updates?samples=1&task=BRAIN_TUMOR', { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(hospitals[1])}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(3, 0, 'densenet121', 'BRAIN_TUMOR') })).response.status, 202);
  for (let attempt = 0; attempt < 40; attempt++) { const ret = await request('/api/federated/rounds/1?task=RETINOPATHY', { headers: { Authorization: `Bearer ${adminToken}` } }); const brainState = await request('/api/federated/rounds/1?task=BRAIN_TUMOR', { headers: { Authorization: `Bearer ${adminToken}` } }); if (ret.body.status === 'COMPLETED' && brainState.body.status === 'COMPLETED') break; await new Promise(resolve => setTimeout(resolve, 10)); }
  const retModel = await request('/api/federated/global-model?task=RETINOPATHY', { headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}` } });
  const brainModel = await request('/api/federated/global-model?task=BRAIN_TUMOR', { headers: { Authorization: `Bearer ${tokens.get(hospitals[1])}` } });
  assert.equal(retModel.body.task, 'RETINOPATHY'); assert.equal(retModel.body.version, 1); assert.equal(brainModel.body.task, 'BRAIN_TUMOR'); assert.equal(brainModel.body.version, 1);
});
test('model downloads require round participation and rollback is task-specific', async () => {
  const unauthorized = await request('/api/federated/global-model?task=PNEUMONIA', { headers: { Authorization: `Bearer ${tokens.get(hospitals[3])}` } });
  assert.equal(unauthorized.response.status, 403);
  const rollback = await request('/api/admin/models/1/rollback?task=PNEUMONIA', json({}, adminToken));
  assert.equal(rollback.response.status, 200);
  const model = await request('/api/federated/global-model?task=PNEUMONIA', { headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}` } });
  assert.equal(model.response.status, 200); assert.equal(model.body.version, 1);
  const retinopathy = await request('/api/federated/global-model?task=RETINOPATHY', { headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}` } });
  assert.equal(retinopathy.response.status, 200); assert.equal(retinopathy.body.version, 1);
});

test('hospital federation responses reveal only the caller’s own participation', async () => {
  const hospitalRound = await request('/api/federated/rounds/1?task=PNEUMONIA', { headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}` } });
  assert.equal(hospitalRound.response.status, 200);
  assert.equal(hospitalRound.body.isParticipant, true);
  assert.equal(hospitalRound.body.hasSubmitted, true);
  assert.equal('participantHospitalIds' in hospitalRound.body, false);
  assert.equal('submittedHospitalIds' in hospitalRound.body, false);
  const overview = await request('/api/hospital/overview?task=PNEUMONIA', { headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}` } });
  assert.equal(overview.response.status, 200);
  assert.equal('participants' in overview.body, false);
  assert.equal('participantHospitalIds' in overview.body.round, false);
});

test('a selected hospital can download the exact base model for its active round', async () => {
  const created = await request('/api/federated/rounds', json({ participantHospitalIds: [hospitals[3]] }, adminToken));
  assert.equal(created.response.status, 201);
  const download = await fetch(baseUrl + `/api/federated/global-model-binary?task=PNEUMONIA&version=${created.body.baseGlobalModelVersion}`, { headers: { Authorization: `Bearer ${tokens.get(hospitals[3])}` } });
  assert.equal(download.status, 200);
  assert.equal(download.headers.get('x-arogyavaani-model-version'), String(created.body.baseGlobalModelVersion));
  await request(`/api/federated/rounds/${created.body.round}/cancel?task=PNEUMONIA`, json({}, adminToken));
});

test('admin retrieves persisted model updates with checksum and artifact status', async () => {
  const updates = await request('/api/admin/model-updates?task=PNEUMONIA', { headers: { Authorization: `Bearer ${adminToken}` } });
  assert.equal(updates.response.status, 200);
  assert.ok(Array.isArray(updates.body));
  assert.ok(updates.body.length > 0);
  const first = updates.body[0];
  assert.equal(first.status, 'stored');
  assert.ok(first.hospitalId);
  assert.ok(first.round);
  assert.ok(first.samples > 0);
});

