/* Offline API-boundary regressions. All fetches are intercepted; no model is run.
 * Run from forge-web with: npm run test:api-contracts
 * This intentionally uses Node's test runner and the existing TypeScript compiler.
 */
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { mkdtempSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');
const { test } = require('node:test');

const temporary = mkdtempSync(path.join(tmpdir(), 'olympus-api-contracts-'));
const source = path.resolve(__dirname, '../src/api.ts');
let compiler = 'tsc';
let prefix = [];
try {
  prefix = [require.resolve('typescript/bin/tsc')];
  compiler = process.execPath;
} catch (error) {
  if (error.code !== 'MODULE_NOT_FOUND') throw error;
}
const compiled = spawnSync(compiler, [...prefix, source,
  '--outDir', temporary, '--module', 'commonjs', '--target', 'ES2022',
  '--lib', 'ES2022,DOM', '--strict', '--skipLibCheck',
  '--typeRoots', path.join(temporary, 'no-ambient-types')
], { encoding: 'utf8', timeout: 30000 });
if (compiled.error || compiled.status !== 0) {
  rmSync(temporary, { recursive: true, force: true });
  throw new Error(`TypeScript compilation failed: ${compiled.error || compiled.stdout + compiled.stderr}`);
}
const api = require(path.join(temporary, 'api.js'));
const status = () => ({ integrity: 'ok', datasets: 1, experiments: 1,
  checkpoints: 1, evaluations: 1, models: 1, evidence_events: 6 });
const model = (id = 'verified-model') => ({ id, object: 'model', owned_by: 'bu1ld-olympus',
  status: 'VERIFIED', capabilities: ['generate'], limitations: ['fixture only'] });
const completion = () => ({ id: 'chatcmpl-fixture', object: 'chat.completion',
  model: 'verified-model', choices: [{ index: 0,
    message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5, unit: 'characters' },
  evidence: { checkpoint_id: 'fixture' } });
const cases = [];
function add(name, invoke, payload, error, options = {}) {
  cases.push({ name, invoke, payload, error, options });
}
const generate = () => api.generateWithFoundry('verified-model', 'hi');

add('status: preserves a valid backend response', api.fetchFoundryStatus, status());
add('status: preserves verified empty counts', api.fetchFoundryStatus,
  { integrity: 'ok', datasets: 0, experiments: 0, checkpoints: 0,
    evaluations: 0, models: 0, evidence_events: 0 });
for (const field of ['datasets', 'experiments', 'checkpoints', 'evaluations', 'models', 'evidence_events']) {
  for (const value of [-1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    add(`status: rejects ${field}=${value}`, api.fetchFoundryStatus,
      { ...status(), [field]: value }, /invalid Foundry status/);
  }
}
for (const [name, invoke, error] of [
  ['status', api.fetchFoundryStatus, /invalid Foundry status/],
  ['registry', api.fetchFoundryModels, /invalid model registry/],
  ['provider', api.fetchOllamaHealth, /invalid Ollama health response/]
]) {
  for (const payload of [null, [], 7, 'ok']) {
    add(`${name}: rejects root ${JSON.stringify(payload)} with the domain error`, invoke, payload, error);
  }
}
add('registry: preserves two distinct model identities', api.fetchFoundryModels,
  { data: [model('model-a'), model('model-b')] });
add('registry: preserves a successful empty list', api.fetchFoundryModels, { data: [] });
add('registry: rejects duplicate identities', api.fetchFoundryModels,
  { data: [model(), model()] }, /invalid model registry/);
for (const id of ['', '   ']) {
  add(`registry: rejects blank identity ${JSON.stringify(id)}`, api.fetchFoundryModels,
    { data: [model(id)] }, /invalid model registry/);
}
add('provider: preserves successful availability', api.fetchOllamaHealth,
  { status: 'ok', provider: 'ollama', models: ['local-model'] });
add('provider: preserves an empty successful list', api.fetchOllamaHealth,
  { status: 'ok', provider: 'ollama', models: [] });
for (const reason of ['stop', 'length']) {
  const payload = completion(); payload.choices[0].finish_reason = reason;
  add(`chat: preserves backend finish reason ${reason}`, generate, payload);
}
const empty = completion(); empty.choices[0].message.content = '';
empty.usage.completion_tokens = 0; empty.usage.total_tokens = 3;
add('chat: preserves a legitimate empty completion', generate, empty);
for (const [name, mutate] of [
  ['wrong model', p => { p.model = 'different-model'; }],
  ['blank receipt id', p => { p.id = '  '; }],
  ['missing finish reason', p => { delete p.choices[0].finish_reason; }],
  ['null finish reason', p => { p.choices[0].finish_reason = null; }],
  ['unknown finish reason', p => { p.choices[0].finish_reason = 'unknown'; }],
  ['missing choice index', p => { delete p.choices[0].index; }],
  ['wrong choice index', p => { p.choices[0].index = 1; }],
  ['extra unexpected choice', p => { p.choices.push({}); }],
  ['inconsistent usage total', p => { p.usage.total_tokens = 99; }],
  ['wrong usage unit', p => { p.usage.unit = 'tokens'; }],
  ['missing evidence', p => { delete p.evidence; }],
  ['non-assistant message', p => { p.choices[0].message.role = 'user'; }]
]) {
  const payload = completion(); mutate(payload);
  add(`chat: rejects ${name}`, generate, payload, /invalid chat completion/);
}
for (const field of ['prompt_tokens', 'completion_tokens', 'total_tokens']) {
  for (const value of [-1, 0.5, Number.MAX_SAFE_INTEGER + 1]) {
    const payload = completion(); payload.usage[field] = value;
    add(`chat: rejects ${field}=${value}`, generate, payload, /invalid chat completion/);
  }
}
add('demo: preserves finite confidence', api.fetchDemos, { forge: { confidence: 0.5, passed: true } });
add('demo: rejects JSON numeric overflow', api.fetchDemos,
  '{"forge":{"confidence":1e309}}', /invalid demo payload/, { raw: true });
add('transport: preserves useful provider error details', api.fetchFoundryStatus,
  { detail: 'Unavailable' }, /API request failed with status 503: Unavailable/, { status: 503 });
add('transport: handles null error payload', api.fetchFoundryStatus,
  null, /API request failed with status 503$/, { status: 503 });
add('transport: does not retry a failed generation mutation', generate,
  { detail: 'Rejected' }, /API request failed with status 422: Rejected/, { status: 422 });

// Serial subtests own and restore fetch, including on assertion failure.
test('offline Foundry API contracts', { concurrency: false }, async (t) => {
  try {
    for (const item of cases) {
      await t.test(item.name, async () => {
        const previousFetch = globalThis.fetch;
        let calls = 0;
        globalThis.fetch = async (input, options) => {
          calls += 1;
          assert.ok(String(input).startsWith('/api/'), 'only same-origin API paths are expected');
          if (item.invoke === generate) {
            assert.equal(options.method, 'POST');
            assert.equal(JSON.parse(options.body).model, 'verified-model');
          }
          return new Response(item.options.raw ? item.payload : JSON.stringify(item.payload), {
            status: item.options.status || 200,
            headers: { 'Content-Type': 'application/json' }
          });
        };
        try {
          if (item.error) await assert.rejects(item.invoke, item.error);
          else {
            const expected = item.invoke === api.fetchFoundryModels ? item.payload.data : item.payload;
            assert.deepEqual(await item.invoke(), expected);
          }
          assert.equal(calls, 1, 'each request must be issued once without hidden retries');
        } finally {
          globalThis.fetch = previousFetch;
        }
      });
    }
    await t.test('transport: preserves caller cancellation and does not retry', async () => {
      const previousFetch = globalThis.fetch;
      const controller = new AbortController();
      controller.abort();
      let calls = 0;
      globalThis.fetch = async (_input, options) => {
        calls += 1;
        assert.equal(options.signal, controller.signal);
        throw new DOMException('Aborted', 'AbortError');
      };
      try {
        await assert.rejects(() => api.fetchFoundryStatus(controller.signal), { name: 'AbortError' });
        assert.equal(calls, 1);
      } finally { globalThis.fetch = previousFetch; }
    });
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
});
