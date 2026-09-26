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
const request = async (path: string, init: RequestInit = {}) => { const response = await fetch(baseUrl + path, init); const body = response.status === 204 ? undefined : await response.json() as any; return { response, body }; };
const json = (body: object, token?: string): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) }, body: JSON.stringify(body) });
const payload = (value: number, baseVersion: number, architecture = 'densenet121') => { const data = Buffer.alloc(4); data.writeFloatLE(value); const manifest = Buffer.from(JSON.stringify({ format: 'arogyavaani-state-v1', architecture, baseVersion, classes: ['negative', 'positive'], tensors: [{ name: 'weight', shape: [1], dtype: 'f32', offset: 0, length: 4 }] })); const header = Buffer.alloc(8); header.write('AVM1'); header.writeUInt32LE(manifest.length, 4); return Buffer.concat([header, manifest, data]); };
const submit = (round: number, email: string, value: number, samples: number, baseVersion = 0, architecture = 'densenet121') => request(`/api/federated/rounds/${round}/updates?samples=${samples}`, { method: 'POST', headers: { Authorization: `Bearer ${tokens.get(email)}`, 'Content-Type': 'application/vnd.arogyavaani.model-v1' }, body: payload(value, baseVersion, architecture) });
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
  const first = await submit(1, hospitals[0], 1, 1); assert.equal(first.response.status, 202); assert.equal(first.body.status, 'WAITING_FOR_UPDATES'); assert.equal(first.body.submittedParticipantCount, 1); assert.equal((await request('/api/federated/global-model-binary', { headers: { Authorization: `Bearer ${tokens.get(hospitals[0])}` } })).response.status, 404);
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
