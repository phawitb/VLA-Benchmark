// Unit tests for the real augmentation controls and outgoing API payloads.
// Run: node --test tests/test_augmentation_ui.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
const script = html.slice(html.indexOf('async function vizShowAugment()'),
                         html.indexOf('async function vizShowRandomSubset()'));

function setup() {
  const elements = new Map();
  for (const id of ['augOcclusionEnabled', 'augCamEnabled', 'augLightEnabled',
    'augRobotEnabled', 'augLangEnabled', 'augTargetEps', 'augNewName', 'augIntensity',
    'augStatus', 'augBtn', 'augProgress', 'augBar', 'augPreviewGrid', 'augPreviewStatus',
    'augOcclusionParams', 'augSummary', 'augMultLabel', 'afOccX', 'afOccY', 'afOccW', 'afOccH']) {
    elements.set(id, {value: '', checked: false, style: {}, innerHTML: '', textContent: ''});
  }
  const el = id => elements.get(id);
  el('augIntensity').value = '3';
  el('augTargetEps').value = '4';
  el('augNewName').value = 'generated';
  el('augOcclusionEnabled').checked = true;
  ['afOccX', 'afOccY', 'afOccW', 'afOccH'].forEach((id, i) => el(id).value = [10, 20, 30, 40][i]);
  const calls = [];
  const state = {progress: 60, status: 'Processing', running: true, error: null};
  const context = vm.createContext({
    window: {_augMode: 'fix', _augPreviewName: 'source'},
    document: {getElementById: id => el(id) || null, querySelectorAll: () => []},
    console, setTimeout: () => 1, clearTimeout() {}, clearInterval() {},
    setInterval: fn => { context.poll = fn; return 1; },
    alert: message => calls.push({alert: message}),
    closeModal: () => { context.closed = true; },
    vizRefreshDatasets: () => { context.refreshed = true; },
    fetch: async (url, options) => {
      calls.push({url, body: options ? JSON.parse(options.body) : null});
      return {json: async () => url.endsWith('/status') ? state :
        {ok: true, previews: [{ep: 0, original: 'original-image', augmented: 'augmented-image'}]}};
    },
  });
  vm.runInContext(script, context);
  return {context, el, calls, state};
}

test('Fix preview and generation send the same normalized occlusion rectangle', async () => {
  const {context: c, calls} = setup();
  await c._augPreviewUpdate();
  await c.vizDoAugment('source');
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, '/api/datasets/source/augment-preview');
  assert.equal(calls[1].url, '/api/datasets/source/augment');
  for (const {body} of calls) {
    assert.equal(body.techniques.occlusion.enabled, true);
    assert.equal(body.mode, 'fix');
    assert.deepEqual(body.fixed_params.light.occlusion, [.1, .2, .3, .4]);
  }
  assert.equal(calls[1].body.techniques.light_quality.enabled, false);
});

test('Random and disabled occlusion payloads respect selected controls', async () => {
  const {context: c, el, calls} = setup();
  c.window._augMode = 'random';
  el('augOcclusionEnabled').checked = false;
  await c._augPreviewUpdate();
  await c.vizDoAugment('source');
  for (const {body} of calls) {
    assert.equal(body.techniques.occlusion.enabled, false);
    assert.equal(body.fixed_params, null);
    assert.equal(body.mode, 'random');
  }
});

test('Occlusion renders saved Fix controls, random ranges, and summary', () => {
  const {context: c, el} = setup();
  c.renderAugParams();
  const controls = el('augOcclusionParams').innerHTML;
  for (const label of ['Position X', 'Position Y', 'Width', 'Height']) assert.ok(controls.includes(label));
  assert.ok(controls.includes('value="10"'));
  c.window._augMode = 'random';
  c.renderAugParams();
  assert.match(el('augOcclusionParams').innerHTML, /10.0.*30.0/);
  c.updateAugSummary(2);
  assert.match(el('augSummary').innerHTML, /Random Occlusion/);
  el('augOcclusionEnabled').checked = false;
  c.renderAugParams();
  assert.equal(el('augOcclusionParams').innerHTML, '');
});

test('Progress polling updates the rendered bar and completes the workflow', async () => {
  const {context: c, el, state} = setup();
  await c.vizDoAugment('source');
  await c.poll();
  assert.equal(el('augBar').style.width, '60%');
  assert.equal(el('augStatus').textContent, 'Processing');
  state.progress = 100;
  state.running = false;
  await c.poll();
  assert.equal(c.closed, true);
  assert.equal(c.refreshed, true);
});
