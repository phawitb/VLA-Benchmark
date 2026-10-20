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
    'augDropoutEnabled', 'augDropoutKeep', 'augDropoutMode', 'augDropoutPercent', 'augDropCam0', 'augDropCam1', 'augDropCam2',
    'augRobotEnabled', 'augLangEnabled', 'augTargetEps', 'augNewName', 'augIntensity',
    'augStatus', 'augBtn', 'augProgress', 'augBar', 'augPreviewGrid', 'augPreviewStatus',
    'augOcclusionParams', 'augSummary', 'augMultLabel', 'afOccX', 'afOccY', 'afOccW', 'afOccH']) {
    elements.set(id, {value: '', checked: false, style: {}, innerHTML: '', textContent: ''});
  }
  const el = id => elements.get(id);
  el('augIntensity').value = '3';
  el('augDropoutPercent').value = '50';
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

test('Multi-camera dropout controls send identical preview and generation choices', async () => {
  const {context: c, el, calls} = setup();
  c.window._augCameraKeys = ['observation.images.top', 'observation.images.wrist', 'observation.images.side'];
  el('augDropoutEnabled').checked = true;
  el('augDropCam0').checked = true;
  el('augDropCam2').checked = true;
  el('augDropoutKeep').value = '1';
  el('augDropoutMode').value = 'cycle';
  await c._augPreviewUpdate();
  await c.vizDoAugment('source');
  assert.equal(calls.length, 2);
  for (const {body} of calls) {
    assert.deepEqual(body.techniques.camera_dropout, {enabled: true,
      cameras: ['observation.images.top', 'observation.images.side'], keep_count: 1, mode: 'cycle', drop_camera_percent: 50});
  }
  c.updateAugSummary(2);
  assert.match(el('augSummary').innerHTML, /Camera Dropout/);
  assert.match(el('augSummary').innerHTML, /2 full.*2 dropout/);
  assert.equal(calls[0].body.target_episodes, 4);
});

test('Intensity sets dropout defaults while unrelated controls preserve a manual percentage', async () => {
  const {context: c, el, calls} = setup();
  for (const [intensity, percent] of [[1, 10], [2, 30], [3, 50], [4, 70], [5, 90]]) {
    el('augIntensity').value = String(intensity);
    c.syncAugDropoutControls(true);
    assert.equal(Number(el('augDropoutPercent').value), percent);
    await c._augPreviewUpdate();
    await c.vizDoAugment('source');
    for (const {body} of calls.slice(-2)) {
      assert.equal(body.intensity, intensity);
      assert.equal(body.techniques.camera_dropout.drop_camera_percent, percent);
    }
  }
  el('augDropoutPercent').value = '65';
  c.syncAugDropoutControls();
  assert.equal(c.getCameraDropout().drop_camera_percent, 65);
  el('augIntensity').value = '2';
  c.syncAugDropoutControls(true);
  assert.equal(c.getCameraDropout().drop_camera_percent, 30);
});

test('Preview renders every camera with its kept or black status', async () => {
  const {context: c, el} = setup();
  c.fetch = async () => ({json: async () => ({ok: true, previews: [{ep: 0,
    cameras: [{camera: 'observation.images.top', dropped: false, original: 'a', augmented: 'b'},
      {camera: 'observation.images.wrist', dropped: true, original: 'c', augmented: 'd'}]}]})});
  await c._augPreviewUpdate();
  assert.match(el('augPreviewGrid').innerHTML, /top/);
  assert.match(el('augPreviewGrid').innerHTML, /wrist/);
  assert.match(el('augPreviewGrid').innerHTML, /Black/);
  assert.match(el('augPreviewGrid').innerHTML, /Active/);
});

test('A slow old preview cannot replace the latest camera selection', async () => {
  const {context: c, el} = setup();
  const pending = [];
  c.fetch = async () => ({json: () => new Promise(resolve => pending.push(resolve))});
  const first = c._augPreviewUpdate();
  await new Promise(setImmediate);
  const latest = c._augPreviewUpdate();
  await new Promise(setImmediate);
  pending[1]({ok: true, previews: [{ep: 2, cameras: [
    {camera: 'wrist', dropped: true, original: 'a', augmented: 'b'}]}]});
  await latest;
  pending[0]({ok: true, previews: [{ep: 1, cameras: [
    {camera: 'top', dropped: false, original: 'a', augmented: 'b'}]}]});
  await first;
  assert.match(el('augPreviewGrid').innerHTML, /wrist/);
  assert.doesNotMatch(el('augPreviewGrid').innerHTML, /top · Active/);
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
