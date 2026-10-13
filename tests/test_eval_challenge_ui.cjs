const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function setup() {
  const elements = new Map();
  const doc = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {value:'',checked:false,style:{},innerHTML:''});
    return elements.get(id);
  }, querySelectorAll() {return [];}};
  const c = vm.createContext({document:doc,evalConfigChanged(){},console,
    AbortController, setTimeout:()=>1, clearTimeout(){}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/eval_challenge.js'),'utf8'),c);
  return {c,doc};
}

test('Save/load retains Fix settings, disabled groups, and every non-language technique',()=>{
  const {c}=setup();
  const value={enabled:true,mode:'fix',intensity:5,
    camera:{enabled:false,perspective:true,affine:true,rotation:true},
    light:{enabled:false,brightness:true,contrast:true,saturation:true,color_jitter:true,shadow:true,noise:true,blur:true},
    occlusion:{enabled:true},robot:{enabled:true,random_start:true,initial_noise:true,trajectory_jitter:true},
    fixed_params:{camera:{angle:-4,tx:.03,ty:-.02,shear:.01,scale:1.05,persp_mag:.04},
      light:{brightness:1.2,contrast:.8,saturation:1.3,hue:6,sat_off:2,val_off:-3,shadow_a:.3,shadow_dir:'right',noise_s:9,blur_s:1.1,occlusion:[.1,.2,.3,.4]},
      robot:{trim:6,offset_sigma:1.1,jitter_s:.7}}};
  c.evalSetChallengeConfig(value);
  const result=JSON.parse(JSON.stringify(c.evalGetChallengeConfig()));
  assert.deepEqual(result,value);
  assert.equal(result.language,undefined);
});

test('Legacy settings default to Random and enabled camera/light without enabling robot',()=>{
  const {c}=setup();
  c.evalSetChallengeConfig({enabled:true,intensity:2,camera:{rotation:true},light:{brightness:true}});
  const r=c.evalGetChallengeConfig();
  assert.equal(r.mode,'random');
  assert.equal(r.camera.enabled,true);
  assert.equal(r.light.enabled,true);
  assert.equal(r.robot.enabled,false);
  assert.equal(r.occlusion.enabled,false);
});

test('Run request includes the complete challenge config used by Save',()=>{
  const {c,doc}=setup();
  c.evalSetChallengeConfig({enabled:true,mode:'fix',occlusion:{enabled:true},robot:{enabled:true,trajectory_jitter:true}});
  c.evalSelectedModel={id:'model',name:'Model'};
  const html=fs.readFileSync(path.join(__dirname,'../static/index.html'),'utf8');
  vm.runInContext(html.slice(html.indexOf('function _evalBuildBody()'),html.indexOf('async function _evalStopProcess()')),c);
  assert.deepEqual(JSON.parse(JSON.stringify(c._evalBuildBody().challenge_env)),JSON.parse(JSON.stringify(c.evalGetChallengeConfig())));
  doc.getElementById('evalChallengeEnabled').checked=false;
  assert.equal(c._evalBuildBody().challenge_env,undefined);
});

test('Preview sends saved Challenge settings and renders original/augmented pairs',async()=>{
  const {c,doc}=setup();
  c.evalSetChallengeConfig({enabled:true,mode:'fix',occlusion:{enabled:true}});
  doc.getElementById('evalChallengePreviewDataset').value='source';
  doc.getElementById('evalChallengePreviewEpisode').value='3';
  let payload;
  c.fetch=async(url,opts)=>{payload=JSON.parse(opts.body);return {ok:true,json:async()=>({ok:true,previews:[{
    source_episode:7,challenge_episode:3,camera:'top',original:'AAA',augmented:'BBB',
    params:{robot:{trim:4,joint_offset:[1,-1],jitter_s:.5}}
  }]})};};
  await c.evalUpdateChallengePreview();
  assert.equal(payload.dataset,'source');
  assert.equal(payload.episode,3);
  assert.equal(payload.challenge_env.mode,'fix');
  const rendered=doc.getElementById('evalChallengePreviewGrid').innerHTML;
  assert.match(rendered,/AAA/);assert.match(rendered,/BBB/);
  assert.match(rendered,/Trim: 4/);assert.match(rendered,/Source ep 7/);
});

test('Preview displays useful errors and ignores stale responses',async()=>{
  const {c,doc}=setup();
  doc.getElementById('evalChallengePreviewDataset').value='source';
  let resolveOld;
  c.fetch=()=>new Promise(resolve=>{resolveOld=resolve;});
  const old=c.evalUpdateChallengePreview();
  c.fetch=async()=>({ok:true,json:async()=>({ok:false,error:'No readable video frames'})});
  await c.evalUpdateChallengePreview();
  resolveOld({ok:true,json:async()=>({ok:true,previews:[{source_episode:0,challenge_episode:1,camera:'top',original:'OLD',augmented:'OLD'}]})});
  await old;
  assert.match(doc.getElementById('evalChallengePreviewStatus').textContent,/No readable video frames/);
  assert.doesNotMatch(doc.getElementById('evalChallengePreviewGrid').innerHTML,/OLD/);
});

test('Setup opens a modal and retains Fix settings when closed and reopened',()=>{
  const {c,doc}=setup();
  const dialog=doc.getElementById('evalChallengeDialog');
  dialog.showModal=()=>{dialog.open=true;};
  dialog.close=()=>{dialog.open=false;c.evalChallengeClosed();};
  let previews=0;
  c.evalLoadChallengePreviewSources=()=>{previews++;};
  c.evalSetChallengeConfig({enabled:true,mode:'fix',fixed_params:{camera:{angle:6}}});
  c.evalOpenChallengeSetup();
  assert.equal(dialog.open,true);
  assert.equal(doc.getElementById('evalChallengeBody').style.display,'');
  c.evalCloseChallengeSetup();
  assert.equal(dialog.open,false);
  assert.equal(doc.getElementById('evalChallengeBody').style.display,'none');
  c.evalOpenChallengeSetup();
  assert.equal(c.evalGetChallengeConfig().fixed_params.camera.angle,6);
  assert.equal(c.evalGetChallengeConfig().mode,'fix');
  assert.equal(previews,2);
});
