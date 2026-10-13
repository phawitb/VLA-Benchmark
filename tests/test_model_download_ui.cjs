const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup() {
  const elements = new Map();
  const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {style:{},textContent:'',innerHTML:'',disabled:false,options:[]});
    return elements.get(id);
  }};
  let tick;
  const model = {id:'user/model',name:'model',local_exists:false};
  const c = vm.createContext({document,AbortSignal,evalSelectedModel:model,evalModels:[model],
    evalSelectModel(){ c.selected = true; },
    evalLoadModels(){ throw new Error('Completion must not depend on remote listing'); },
    setInterval(fn){tick=fn; return 1;},clearInterval(){},setTimeout(){},
    fetch:async()=>({ok:true,json:async()=>({ok:true})})});
  const html=fs.readFileSync('static/index.html','utf8');
  vm.runInContext(html.slice(html.indexOf('let evalDownloadPollTimer'),html.indexOf('// ── Test Inference')),c);
  return {c,document,tick:()=>tick()};
}

test('Completion immediately makes the downloaded model usable without a remote reload',async()=>{
  const {c,document,tick}=setup();
  await c.evalDownloadModel();
  c.fetch=async()=>({ok:true,json:async()=>({running:false,status:'Download complete',error:null})});
  await tick();
  assert.equal(c.evalSelectedModel.local_exists,true);
  assert.equal(c.selected,true);
  assert.equal(document.getElementById('evalDownloadBtn').disabled,false);
});

test('Status requests cannot overlap and regress a completed download',async()=>{
  const {c,tick}=setup();
  await c.evalDownloadModel();
  let resolve, calls=0;
  c.fetch=()=>{calls++; return new Promise(r=>{resolve=r;});};
  const pending=tick();
  const overlapping=tick();
  assert.equal(calls,1);
  resolve({ok:true,json:async()=>({running:false,status:'Download complete',error:null})});
  await pending;
  await overlapping;
  await tick();
  assert.equal(calls,1);
});

test('Polling errors are visible and the next poll can recover',async()=>{
  const {c,document,tick}=setup();
  await c.evalDownloadModel();
  c.fetch=async()=>{throw new Error('Network unavailable');};
  await tick();
  assert.match(document.getElementById('evalDownloadStatus').textContent,/Network unavailable/);
  c.fetch=async()=>({ok:true,json:async()=>({running:false,error:null})});
  await tick();
  assert.equal(c.evalSelectedModel.local_exists,true);
});
