const test = require('node:test');
const assert = require('node:assert/strict');
const {validate,progress,requestGate,safeURL,esc} = require('../static/libero_eval.js');
const config = () => ({model_path:'org/policy',suite:'libero_spatial',task_ids:[0,2],episodes:10,max_steps:300,seed:0,device:'auto'});
test('valid benchmark settings preserve selected task IDs',()=>assert.deepEqual(validate(config()).task_ids,[0,2]));
test('invalid input cannot silently create a different benchmark',()=>{
  for(const patch of [{model_path:' '},{task_ids:[]},{task_ids:[NaN]},{episodes:1.5},{episodes:0},{max_steps:Infinity},{seed:NaN},{seed:-1},{device:'gpu'}]) assert.throws(()=>validate({...config(),...patch}));
});
test('progress is episode-based and bounded including cancelled and idle runs',()=>{
  assert.deepEqual(progress(null),{total:0,completed:0,percent:0});
  assert.deepEqual(progress({total:20,completed:7,status:'cancelled'}),{total:20,completed:7,percent:35});
  assert.equal(progress({total:10,completed:50}).percent,100);
  assert.equal(progress({total:10,completed:-3}).percent,0);
});
test('late responses cannot replace a newer selected history detail or status',async()=>{
  const gate=requestGate();let displayed;
  let resolveOld;const oldResponse=new Promise(resolve=>resolveOld=resolve);
  const oldToken=gate.next();const old=oldResponse.then(value=>{if(gate.current(oldToken))displayed=value;});
  const newToken=gate.next();if(gate.current(newToken))displayed='new run';
  resolveOld('old run');await old;assert.equal(displayed,'new run');
});
test('untrusted names and media references cannot inject executable markup',()=>{
  assert.equal(esc('<img src=x onerror="evil()">'),'&lt;img src=x onerror=&quot;evil()&quot;&gt;');
  for(const url of ['javascript:alert(1)','//evil.test/x','/\\evil.test/x','https://evil.test/x','/media/\nx'])assert.equal(safeURL(url),'');
  assert.equal(safeURL('/api/libero/video/run/1.mp4'),'/api/libero/video/run/1.mp4');
});

test('HF choices retain repository IDs and reuse known downloaded checkpoints',()=>{
  const {hfModelOptions}=require('../static/libero_eval.js');
  const choices=hfModelOptions({models:[
    {id:'user/remote',name:'remote',local_exists:false},
    {id:'user/local',name:'local',local_exists:true},
  ]},[{name:'local',path:'/models/local',compatible:false,reason:'Requires 7 actions'}]);
  assert.equal(choices[0].path,'/models/local');
  assert.equal(choices[0].compatible,false);
  assert.equal(choices[1].path,'user/remote');
  assert.equal(choices[1].local_exists,false);
  assert.notEqual(choices[1].compatible,true);
});
