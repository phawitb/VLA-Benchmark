/* LIBERO simulation workspace; deliberately independent from physical-robot evaluation. */
(function (root) {
  'use strict';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const active = run => ['starting', 'running'].includes(run?.status);
  function progress(run) {
    const total = Math.max(0, Number(run?.total) || 0);
    const completed = Math.min(total, Math.max(0, Number(run?.completed) || 0));
    return {total, completed, percent: total ? Math.round(completed / total * 100) : 0};
  }
  function validate(config) {
    if (!config.model_path.trim()) throw new Error('Enter a compatible model path or Hugging Face model ID.');
    if (!config.suite || !config.task_ids.length) throw new Error('Select a suite and at least one task.');
    for (const [key, min, max] of [['episodes',1,1000],['max_steps',1,5000],['seed',0,2147483647]]) {
      if (!Number.isInteger(config[key]) || config[key] < min || config[key] > max) throw new Error(`${key.replace('_',' ')} must be a whole number from ${min} to ${max}.`);
    }
    if (!config.task_ids.every(id => Number.isInteger(id) && id >= 0)) throw new Error('Invalid task selection. Reload the catalog.');
    if (!['auto','cpu','cuda','mps'].includes(config.device)) throw new Error('Choose a supported device.');
    return config;
  }
  function requestGate() { let version = 0; return {next: () => ++version, current: token => token === version}; }
  function safeURL(value) {
    if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//') || /[\\\x00-\x20]/.test(value)) return '';
    return value;
  }
  function hfModelOptions(response, localModels = []) {
    return (response.models || []).map(model => {
      const local = model.local_exists ? localModels.find(item => item.name === model.name) : null;
      return {id:model.id || model.repo_id, name:model.name, local_exists:!!model.local_exists,
        path:local?.path || model.id || model.repo_id,
        compatible:local?.compatible, reason:local?.reason || ''};
    }).sort((a,b)=>a.name.localeCompare(b.name));
  }
  const api = {progress, validate, requestGate, safeURL, esc, hfModelOptions};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof document === 'undefined') return;
  const page = document.getElementById('page-libero-evaluate');
  if (!page) return;
  const $ = id => document.getElementById('libero' + id);
  const groups = Object.fromEntries(Object.entries(evalChallengeGroups)
    .filter(([key])=>key!=='robot')
    .map(([key,spec])=>[key,{title:spec.title,enabled:spec.enabled,options:spec.options.map(o=>o[0])}]));
  const fields = typeof evalChallengeFields !== 'undefined' ? evalChallengeFields.filter(f => f[0] !== 'robot').map(f => [...f]) : [];
  const label = key => key.replaceAll('_',' ');
  let catalog = null, liveRun = null, viewedRun = null, timer = null, pending = false, deletedRunIds = [], historyBusy = false, historyRuns = [], historySelecting = false;
  const selectedHistory = new Set();
  const statusGate = requestGate(), historyGate = requestGate(), detailGate = requestGate(), catalogGate = requestGate();
  page.innerHTML = `
    <div class="page-title"><span>Libero</span>-Evaluate</div>
    <p class="libero-intro">Evaluate language-conditioned manipulation in simulation. Compare suites, tasks and repeatable visual challenges.</p>
    <div class="libero-tabs" role="tablist" aria-label="LIBERO workspace">
      <button type="button" id="liberoRunTab" role="tab" aria-selected="true" aria-controls="liberoRunPanel">Run</button>
      <button type="button" id="liberoHistoryTab" role="tab" aria-selected="false" aria-controls="liberoHistoryPanel">History</button>
    </div>
    <div id="liberoMessage" role="status" aria-live="polite"></div>
    <div id="liberoRunPanel" role="tabpanel" aria-labelledby="liberoRunTab" class="libero-layout">
      <form id="liberoForm" class="libero-card">
        <h2>Evaluation setup</h2>
        <div id="liberoRuntime" class="libero-note">Checking simulation runtime…</div>
        <label>Available models<select id="liberoModels"><option value="">Choose a model or enter its path below</option></select></label>
        <div id="liberoModelSource" class="libero-note">Loading models from Hugging Face…</div>
        <label>Model path / Hugging Face ID<input id="liberoModel" placeholder="organization/model or /path/to/checkpoint" required></label>
        <label>Benchmark suite<select id="liberoSuite"></select></label>
        <div class="libero-toolbar"><strong>Tasks <span id="liberoTaskCount"></span></strong><button type="button" id="liberoAll">Select all</button><button type="button" id="liberoNone">Clear</button></div>
        <div id="liberoTasks" class="libero-tasks"></div>
        <div class="libero-grid">
          <label>Episodes per task<input id="liberoEpisodes" type="number" min="1" max="1000" value="10" required></label>
          <label>Max steps per episode<input id="liberoSteps" type="number" min="1" max="5000" value="300" required></label>
          <label>Seed<input id="liberoSeed" type="number" min="0" max="2147483647" value="0" required></label>
          <label>Device<select id="liberoDevice"><option>auto</option><option>cpu</option><option>cuda</option><option>mps</option></select></label>
        </div>
        <label class="libero-check"><input id="liberoVideo" type="checkbox" checked> Save episode videos</label>
        <div class="libero-challenge"><div class="libero-challenge-row"><label class="libero-check"><input id="liberoChallenge" type="checkbox"> Enable challenge</label><button type="button" id="liberoChallengeSetup">Setup</button></div>
        <dialog id="liberoChallengeDialog" aria-labelledby="liberoChallengeTitle">
          <header class="libero-dialog-header"><h2 id="liberoChallengeTitle">Challenge Environment</h2><div class="libero-dialog-actions"><label class="libero-check"><input id="liberoChallengeDialogEnabled" type="checkbox"><span>Enable challenge</span></label><button type="button" id="liberoChallengeClose">Done</button></div></header>
          <div class="eval-challenge-layout"><section class="eval-challenge-settings"><h3>Settings</h3>
          <div id="liberoChallengeControls">
            <div class="libero-grid"><label>Mode<select id="liberoMode" hidden><option value="random">Random · seeded per episode</option><option value="fix">Fixed parameters</option></select><span class="libero-mode-buttons"><button type="button" id="liberoModeRandom" class="eval-challenge-mode">Random</button><button type="button" id="liberoModeFix" class="eval-challenge-mode">Fix</button></span></label>
            <label id="liberoIntensityRow">Intensity<input id="liberoIntensity" type="range" min="1" max="5" value="3"><output id="liberoIntensityValue">3</output></label></div>
            ${Object.entries(groups).map(([g,s]) => `<section class="libero-tech-card" role="group" aria-labelledby="liberoGroupTitle_${g}"><label class="libero-check libero-tech-heading"><input type="checkbox" id="liberoGroup_${g}" ${s.enabled?'checked':''}><span id="liberoGroupTitle_${g}">${s.title}</span></label>
              <div class="libero-tech-options">${s.options.map(k=>`<label class="libero-check libero-tech-option"><input type="checkbox" id="liberoOption_${g}_${k}" checked><span>${esc(evalChallengeGroups[g].options.find(o=>o[0]===k)?.[1]||label(k))}</span></label>`).join('')}</div>
              <div class="eval-challenge-params" id="liberoRandom_${g}"></div><div data-libero-fixed="${g}">${fields.filter(f=>f[0]===g).map(([group,key,title,min,max,step,value,tech])=>`<label class="eval-challenge-slider" data-libero-tech="${tech}"><span class="eval-challenge-slider-label">${esc(title)}</span><input type="range" id="liberoFixed_${group}_${key}" min="${min}" max="${max}" step="${step}" value="${value}"><output id="liberoFixed_${group}_${key}Value">${value}</output></label>`).join('')}
              ${g==='light'?'<label>Shadow direction<select id="liberoShadow"><option>left</option><option>right</option><option>top</option><option>bottom</option></select></label>':''}</div></section>`).join('')}
          </div>
          </section><section class="eval-challenge-preview"><h3>Preview</h3>
          <label>LIBERO task<select id="liberoPreviewTask"></select></label>
          <p id="liberoPreviewStatus" role="status">Select a LIBERO task to preview.</p><div id="liberoPreviewGrid"></div>
          </section></div></dialog></div>
        <p id="liberoEstimate" class="libero-note"></p>
        <button type="submit" id="liberoStart" class="libero-primary" disabled>Start evaluation</button>
      </form>
      <section class="libero-card libero-monitor" aria-label="Live evaluation">
        <div class="libero-toolbar"><h2>Live evaluation</h2><button type="button" id="liberoRefresh">Refresh</button><button type="button" id="liberoStop" disabled>Cancel run</button></div>
        <div id="liberoLive"><p class="libero-note">No evaluation has started.</p></div>
      </section>
    </div>
    <section id="liberoHistoryPanel" role="tabpanel" aria-labelledby="liberoHistoryTab" hidden>
      <div class="libero-toolbar"><h2>Evaluation history</h2><button type="button" id="liberoHistoryRefresh">Refresh history</button></div>
      <div class="libero-history-layout"><div class="libero-card libero-history-sidebar"><div class="libero-history-actions"><strong>Saved runs</strong><button type="button" id="liberoHistorySelect">Select</button><label id="liberoHistorySelectAllLabel" class="libero-check" hidden><input type="checkbox" id="liberoHistorySelectAll"> Select all</label><button type="button" id="liberoDelete" hidden disabled>Delete selected (0)</button><button type="button" id="liberoUndo" hidden>Undo delete</button></div><div id="liberoHistoryList">Loading history…</div></div><div class="libero-card"><div class="libero-toolbar"><button type="button" id="liberoReload" disabled>Load configuration</button><button type="button" id="liberoExport" disabled>Export JSON</button></div><div id="liberoDetail"><div class="libero-empty-state"><strong>Select a saved run</strong><p>View results, videos and challenge settings here.</p></div></div></div></div>
    </section>`;
  async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }
  function message(text, error = false) { $('Message').textContent = text; $('Message').className = error ? 'libero-error' : 'libero-note'; }
  function syncButtons() { $('Start').disabled = pending || active(liveRun) || !catalog?.runtime?.available; $('Stop').disabled = pending || !active(liveRun); }
  function renderTasks(selected) {
    const suite = catalog?.suites?.find(s=>s.id===$('Suite').value);
    $('Tasks').innerHTML = (suite?.tasks || []).map(t=>`<label class="libero-check"><input type="checkbox" data-libero-task="${esc(t.id)}" ${!selected || selected.includes(Number(t.id))?'checked':''}>${esc(t.description || t.name.replaceAll('_',' '))}</label>`).join('');
    estimate();
  }
  function selectedTasks() { return Array.from($('Tasks').querySelectorAll('input:checked'), el=>Number(el.dataset.liberoTask)); }
  function estimate() { const n=selectedTasks().length; $('TaskCount').textContent=`(${n} selected)`; $('Estimate').textContent=`${n} tasks × ${$('Episodes').value || 0} episodes = ${n * Number($('Episodes').value || 0)} total episodes`; }
  function syncChallenge() {
    $('ChallengeDialogEnabled').checked=$('Challenge').checked;
    $('ChallengeControls').hidden = false;
    for(const [g,k] of fields) $('Fixed_'+g+'_'+k+'Value').textContent=$('Fixed_'+g+'_'+k).value;
    $('IntensityValue').textContent = $('Intensity').value;
    for(const [g,s]of Object.entries(groups)) {const el=$('Random_'+g);el.hidden=$('Mode').value!=='random'||!$('Group_'+g).checked;el.innerHTML=challengeRandomRanges(g,g==='occlusion'?['occlusion']:s.options.filter(k=>$('Option_'+g+'_'+k).checked),Number($('Intensity').value),Number($('Steps').value));}
    $('IntensityRow').hidden = $('Mode').value === 'fix';
    $('ModeRandom').setAttribute('aria-pressed',String($('Mode').value==='random'));
    $('ModeFix').setAttribute('aria-pressed',String($('Mode').value==='fix'));
    page.querySelectorAll('[data-libero-fixed]').forEach(el => {
      const g = el.dataset.liberoFixed;
      el.hidden = $('Mode').value !== 'fix' || !$('Group_'+g).checked;
      el.querySelectorAll('[data-libero-tech]').forEach(row=> { row.hidden=!!row.dataset.liberoTech && !$('Option_'+g+'_'+row.dataset.liberoTech)?.checked; row.querySelectorAll('input,select').forEach(input=>input.disabled=el.hidden||row.hidden); });
    });
  }
  function challengeConfig() {
    const c={enabled:$('Challenge').checked,mode:$('Mode').value,intensity:Number($('Intensity').value),fixed_params:{camera:{},light:{},robot:{}}};
    for (const [g,s] of Object.entries(groups)) { c[g]={enabled:$('Group_'+g).checked}; for(const k of s.options)c[g][k]=$('Option_'+g+'_'+k).checked; }
    const occ={};
    for(const [g,k,,min,max,,fallback] of fields) {
      const input=$('Fixed_'+g+'_'+k);
      const raw=input.value;
      const value=input.disabled||raw===''?fallback:Number(raw);
      if(!Number.isFinite(value)||value<min||value>max) throw new Error(`Challenge ${label(k)} must be between ${min} and ${max}.`);
      if(g==='occlusion')occ[k]=value/100;else c.fixed_params[g][k]=value;
    }
    c.fixed_params.light.shadow_dir=$('Shadow')?.value || 'left';
    c.fixed_params.light.occlusion=['x','y','width','height'].map(k=>occ[k]);
    return c;
  }
  function config() { return validate({model_path:$('Model').value.trim(),suite:$('Suite').value,task_ids:selectedTasks(),episodes:Number($('Episodes').value),max_steps:Number($('Steps').value),seed:$('Seed').value===''?NaN:Number($('Seed').value),device:$('Device').value,save_video:$('Video').checked,challenge_env:challengeConfig()}); }
  function applyConfig(c) {
    $('Model').value=c.model_path || ''; $('Suite').value=c.suite || catalog?.suites?.[0]?.id || ''; renderTasks(c.task_ids);
    for(const [id,key,fallback] of [['Episodes','episodes',10],['Steps','max_steps',300],['Seed','seed',0],['Device','device','auto']])$(id).value=c[key]??fallback;
    $('Video').checked=c.save_video??true;
    const ch=c.challenge_env || {}; $('Challenge').checked=!!ch.enabled; $('Mode').value=ch.mode==='fix'?'fix':'random'; $('Intensity').value=ch.intensity??3;
    for(const [g,s]of Object.entries(groups)){ $('Group_'+g).checked=ch[g]?.enabled??s.enabled; s.options.forEach((k,i)=>$('Option_'+g+'_'+k).checked=ch[g]?.[k]??true); }
    for(const [g,k,,,,,fallback]of fields){ const oi=['x','y','width','height'].indexOf(k); const value=g==='occlusion'?ch.fixed_params?.light?.occlusion?.[oi]*100:ch.fixed_params?.[g]?.[k]; $('Fixed_'+g+'_'+k).value=Number.isFinite(value)?value:fallback; }
    if($('Shadow'))$('Shadow').value=ch.fixed_params?.light?.shadow_dir || 'left';
    syncChallenge(); estimate();
  }
  function renderRun(run, target) {
    if(!run){ target.innerHTML='<div class="libero-empty-state"><strong>Ready to evaluate</strong><p>Select a model and tasks, then start evaluation.</p><span>Progress, camera previews and results will appear here.</span></div>'; return; }
    const p=progress(run), episodes=Array.isArray(run.episodes)?run.episodes:[], taskStats=new Map();
    episodes.filter(e=>!e.cancelled).forEach(e=>{const key=String(e.task_id);const s=taskStats.get(key)||{name:e.task_name || key,n:0,wins:0};s.n++;s.wins+=e.success?1:0;taskStats.set(key,s);});
    const successes=Math.max(0,Number(run.successes)||0), rate=p.completed?Math.round(successes/p.completed*100):null;
    const preview=safeURL(run.preview_url);
    target.innerHTML=`<div class="libero-toolbar"><strong>${esc(run.status)}</strong><small>${esc(run.id)}</small></div>
      <p class="libero-note">${esc(run.config?.suite || '')} · ${esc(run.config?.model_path || '')}</p>
      <div class="libero-metrics"><div><strong>${p.completed} / ${p.total}</strong><span>Episodes complete</span></div><div><strong>${rate===null?'—':rate+'%'}</strong><span>Success rate · ${successes} successes</span></div><div><strong>${Math.round(Math.max(0,Number(run.elapsed_seconds)||0))}s</strong><span>Elapsed</span></div></div>
      <progress max="100" value="${p.percent}" aria-label="Completed episodes">${p.percent}%</progress>
      <p>${esc(run.current_task ?? 'Waiting for task')} · Episode ${esc(run.current_episode??'—')} · Step ${esc(run.current_step??0)} / ${esc(run.max_steps??run.config?.max_steps??'—')}</p>
      ${run.error?`<p class="libero-error">${esc(run.error)}</p>`:''}
      ${preview?`<img class="libero-preview" alt="Policy input after image challenges" src="${esc(preview)}">`:''}
      <details open><summary>Task results</summary><div class="libero-table-scroll"><table><thead><tr><th>Task</th><th>Episodes</th><th>Successes</th><th>Success rate</th></tr></thead><tbody>${Array.from(taskStats.values(),s=>`<tr><td>${esc(s.name)}</td><td>${s.n}</td><td>${s.wins}</td><td>${Math.round(s.wins/s.n*100)}%</td></tr>`).join('') || '<tr><td colspan="4">No completed episodes yet.</td></tr>'}</tbody></table></div></details>
      <details><summary>Episode results (${episodes.length})</summary><div class="libero-table-scroll"><table><thead><tr><th>Task / episode</th><th>Outcome</th><th>Steps</th><th>Time</th><th>Video</th></tr></thead><tbody>${episodes.map(e=>`<tr><td>${esc(e.task_name || e.task_id)} / ${esc(e.episode)}</td><td>${e.cancelled?'Cancelled':e.success?'Success':'Unsuccessful'}</td><td>${esc(e.steps)}</td><td>${esc(e.duration_seconds)}s</td><td>${safeURL(e.video_url)?`<a href="${esc(safeURL(e.video_url))}" target="_blank" rel="noopener">Watch</a>`:'—'}</td></tr>`).join('')}</tbody></table></div></details>
      <details><summary>Applied challenge parameters · per episode</summary><pre>${esc(JSON.stringify(episodes.map(e=>({task:e.task_id,episode:e.episode,parameters:e.challenge_params||{}})),null,2))}</pre></details><details><summary>Run configuration</summary><pre>${esc(JSON.stringify(run.config || {},null,2))}</pre></details>
      <details><summary>Logs</summary><pre>${esc((run.logs || []).join('\n') || 'No logs yet.')}</pre></details>`;
  }
  async function poll() {
    clearTimeout(timer); const token=statusGate.next();
    try { const data=await fetchJSON('/api/libero/status'); if(!statusGate.current(token))return;
      const wasActive=active(liveRun); liveRun=data.run;
      const open=Array.from($('Live').querySelectorAll('details'),d=>d.open);renderRun(liveRun,$('Live'));$('Live').querySelectorAll('details').forEach((d,i)=>{if(open[i]!==undefined)d.open=open[i];});
      syncButtons(); if(wasActive&&!active(liveRun)){loadHistory();message(liveRun?.status==='completed'?'Evaluation complete.':liveRun?.error||'Evaluation '+(liveRun?.status||'stopped'),liveRun?.status==='failed');}
    } catch(e){if(statusGate.current(token))message(e.message,true);}
    finally { if(statusGate.current(token))timer=setTimeout(poll,active(liveRun)?1500:5000); }
  }
  async function loadCatalog() {
    const token=catalogGate.next();
    try { const [catalogResult, modelsResult] = await Promise.allSettled([
        fetchJSON('/api/libero/catalog'), fetchJSON('/api/hf-models')]);
      if(!catalogGate.current(token))return;
      if(catalogResult.status==='rejected')throw catalogResult.reason;
      const data=catalogResult.value;
      if(modelsResult.status==='fulfilled') {
        const hf=modelsResult.value;
        data.models=hfModelOptions(hf,data.models);
        $('ModelSource').textContent=hf.warning || (hf.hf_user ? 'Hugging Face · '+hf.hf_user+' · '+data.models.length+' models' : 'Set your Hugging Face account in Setup to list models.');
      } else {
        $('ModelSource').textContent='Hugging Face unavailable; showing local checkpoints. Use Refresh to retry.';
      }
      const first=!catalog; catalog=data;
      const suite=$('Suite').value,selected=selectedTasks();
      $('Suite').innerHTML=(data.suites||[]).map(s=>`<option value="${esc(s.id)}">${esc(s.name)} (${s.tasks?.length||0} tasks)</option>`).join('');
      if((data.suites||[]).some(s=>s.id===suite))$('Suite').value=suite;
      $('Models').innerHTML='<option value="">Choose a model or enter its path below</option>'+(data.models||[]).map(m=>`<option value="${esc(m.path || m.id)}" ${m.compatible===false?'disabled':''}>${esc(m.name || m.id)}${m.local_exists===false?' (remote)':''}${m.compatible===false?' — '+esc(m.reason || 'incompatible'):''}</option>`).join('');
      $('Runtime').textContent=data.runtime?.available?'Simulation runtime ready':data.runtime?.error || 'LIBERO runtime unavailable. Install the simulation dependencies before starting.';
      $('Runtime').className=data.runtime?.available?'libero-note':'libero-error';
      renderTasks(first?undefined:selected);syncButtons();
    }catch(e){if(catalogGate.current(token)){message(e.message,true);$('Runtime').textContent='Unable to check runtime. Use Refresh to retry.';}}
  }
  function syncHistorySelection(){
    const eligible=historyRuns.filter(r=>!active(r));
    $('HistorySelect').textContent=historySelecting?'Cancel':'Select';
    $('HistorySelect').disabled=historyBusy||!historyRuns.length;
    $('HistorySelectAllLabel').hidden=!historySelecting;
    $('Delete').hidden=!historySelecting;
    $('HistorySelectAll').checked=eligible.length>0&&eligible.every(r=>selectedHistory.has(r.id));
    $('HistorySelectAll').indeterminate=selectedHistory.size>0&&!$('HistorySelectAll').checked;
    $('HistorySelectAll').disabled=historyBusy||!eligible.length;
    $('Delete').disabled=historyBusy||!selectedHistory.size;
    $('Delete').textContent=`Delete selected (${selectedHistory.size})`;
    $('Undo').hidden=!deletedRunIds.length;$('Undo').disabled=historyBusy;
    $('HistoryList').querySelectorAll('[data-libero-select]').forEach(el=>{el.hidden=!historySelecting;el.checked=selectedHistory.has(el.dataset.liberoSelect);el.disabled=historyBusy||active(historyRuns.find(r=>r.id===el.dataset.liberoSelect));});
  }
  async function loadHistory() {
    const token=historyGate.next();
    try {const data=await fetchJSON('/api/libero/history');if(!historyGate.current(token))return;
      historyRuns=data.runs||[];
      for(const id of selectedHistory)if(!historyRuns.some(r=>r.id===id&&!active(r)))selectedHistory.delete(id);
      $('HistoryList').innerHTML=historyRuns.map(r=>`<div class="libero-history-row"><input type="checkbox" data-libero-select="${esc(r.id)}" hidden aria-label="Select ${esc(r.config?.suite||'run')} ${esc(r.id)}" ${active(r)?'disabled':''}><button type="button" class="libero-history-item" data-libero-run="${esc(r.id)}"><strong>${esc(r.config?.suite || r.suite || r.id)}</strong><span>${esc(r.status)} · ${progress(r).completed}/${progress(r).total} episodes</span><small>${esc(r.config?.model_path?.split('/').pop()||'No model')}</small><small>${esc(r.created_at?new Date(r.created_at).toLocaleString():r.id.slice(0,8))}</small></button></div>`).join('') || '<p>No saved evaluations yet.</p>';
      syncHistorySelection();
    }catch(e){if(historyGate.current(token))$('HistoryList').textContent=e.message;}
  }
  async function showDetail(id) {
    const token=detailGate.next(); viewedRun=null;$('Reload').disabled=true;$('Export').disabled=true;$('Detail').textContent='Loading evaluation…';
    try {const data=await fetchJSON('/api/libero/history/'+encodeURIComponent(id));if(!detailGate.current(token))return;viewedRun=data.run;renderRun(viewedRun,$('Detail'));$('Reload').disabled=!viewedRun?.config;$('Export').disabled=!viewedRun;}
    catch(e){if(detailGate.current(token))$('Detail').textContent=e.message;}
  }
  function tab(history) { $('RunPanel').hidden=history;$('HistoryPanel').hidden=!history;$('RunTab').setAttribute('aria-selected',String(!history));$('HistoryTab').setAttribute('aria-selected',String(history));if(history)loadHistory(); }
  $('RunTab').addEventListener('click',()=>tab(false));$('HistoryTab').addEventListener('click',()=>tab(true));
  $('Models').addEventListener('change',()=>{if($('Models').value)$('Model').value=$('Models').value;});
  $('Suite').addEventListener('change',()=>renderTasks());
  for(const [id,checked]of [['All',true],['None',false]])$(id).addEventListener('click',()=>{$('Tasks').querySelectorAll('input').forEach(el=>el.checked=checked);estimate();});
  $('Form').addEventListener('input',()=>{estimate();syncChallenge();});$('Form').addEventListener('change',syncChallenge);
  $('Form').addEventListener('submit',async event=>{
    event.preventDefault();if(pending||active(liveRun)||!catalog?.runtime?.available)return;
    try {const payload=config();pending=true;syncButtons();statusGate.next();clearTimeout(timer);message('Starting evaluation…');
      const data=await fetchJSON('/api/libero/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      liveRun={id:data.run_id,status:'starting',config:payload,total:payload.task_ids.length*payload.episodes,completed:0};renderRun(liveRun,$('Live'));message('Evaluation started.');
    }catch(e){message(e.message,true);}finally{pending=false;syncButtons();poll();}
  });
  $('Stop').addEventListener('click',async()=>{if(pending)return;pending=true;syncButtons();try{await fetchJSON('/api/libero/stop',{method:'POST'});message('Cancellation requested. Waiting for the current operation to stop.');}catch(e){message(e.message,true);}finally{pending=false;syncButtons();poll();}});
  $('Refresh').addEventListener('click',()=>{loadCatalog();poll();});$('HistoryRefresh').addEventListener('click',loadHistory);
  $('HistoryList').addEventListener('click',event=>{const button=event.target.closest('[data-libero-run]');if(button)showDetail(button.dataset.liberoRun);});
  $('HistorySelect').addEventListener('click',()=>{if(historyBusy)return;historySelecting=!historySelecting;selectedHistory.clear();syncHistorySelection();});
  $('HistoryList').addEventListener('change',event=>{const id=event.target.dataset.liberoSelect;if(!id||historyBusy)return;event.target.checked?selectedHistory.add(id):selectedHistory.delete(id);syncHistorySelection();});
  $('HistorySelectAll').addEventListener('change',()=>{if(historyBusy)return;selectedHistory.clear();if($('HistorySelectAll').checked)historyRuns.filter(r=>!active(r)).forEach(r=>selectedHistory.add(r.id));syncHistorySelection();});
  $('Delete').addEventListener('click',async()=>{
    if(historyBusy||!selectedHistory.size)return;
    historyBusy=true;syncHistorySelection();const removed=[],errors=[];
    for(const id of [...selectedHistory]){
      try{await fetchJSON('/api/libero/history/'+encodeURIComponent(id),{method:'DELETE'});removed.push(id);selectedHistory.delete(id);
        if(viewedRun?.id===id){detailGate.next();viewedRun=null;$('Reload').disabled=true;$('Export').disabled=true;$('Detail').textContent='History deleted. Use Undo delete to restore its results and videos.';}
      }catch(e){errors.push(e.message);}
    }
    deletedRunIds.push(...removed);historyBusy=false;if(!selectedHistory.size)historySelecting=false;
    message(`${removed.length} histories deleted.`+(errors.length?' '+errors.join(' · '):''),!!errors.length);
    await loadHistory();syncHistorySelection();poll();
  });
  $('Undo').addEventListener('click',async()=>{
    if(historyBusy||!deletedRunIds.length)return;
    historyBusy=true;syncHistorySelection();const failed=[],errors=[];let restored=0;
    for(const id of deletedRunIds){try{await fetchJSON('/api/libero/history/'+encodeURIComponent(id)+'/restore',{method:'POST'});restored++;}catch(e){failed.push(id);errors.push(e.message);}}
    deletedRunIds=failed;historyBusy=false;
    message(`${restored} histories restored.`+(errors.length?' '+errors.join(' · '):''),!!errors.length);
    await loadHistory();syncHistorySelection();
  });
  $('Reload').addEventListener('click',()=>{if(!viewedRun?.config)return;applyConfig(viewedRun.config);tab(false);message('Saved configuration loaded. Review the settings before starting a new run.');});
  $('Export').addEventListener('click',()=>{if(!viewedRun)return;const url=URL.createObjectURL(new Blob([JSON.stringify(viewedRun,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='libero-'+String(viewedRun.id).replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  let previewTimer, previewRequest=0, previewAbort;
  async function updatePreview(){
    const request=++previewRequest;previewAbort?.abort();previewAbort=new AbortController();
    const task=$('PreviewTask').value;
    if(task===''){$('PreviewGrid').innerHTML='';return;}
    $('PreviewStatus').textContent='Generating preview…';
    try{
      const data=await fetchJSON('/api/libero/preview',{method:'POST',signal:previewAbort.signal,
        headers:{'Content-Type':'application/json'},body:JSON.stringify({suite:$('Suite').value,task_id:Number(task),seed:Number($('Seed').value),max_steps:Number($('Steps').value),challenge_env:challengeConfig()})});
      if(request!==previewRequest)return;
      $('PreviewGrid').innerHTML=(data.previews||[]).map(p=>`<article><small>${esc(p.task_name)} · ${esc(p.camera)}</small><div class="libero-preview-pair"><figure><img alt="Original" src="data:image/png;base64,${esc(p.original)}"><figcaption>Original</figcaption></figure><figure><img alt="Challenge preview" src="data:image/png;base64,${esc(p.augmented)}"><figcaption>Challenge</figcaption></figure></div></article>`).join('');
      $('PreviewStatus').textContent=data.previews?.length?'':'No preview images available.';
    }catch(e){if(request===previewRequest && e.name!=='AbortError'){$('PreviewStatus').textContent=e.message;$('PreviewGrid').innerHTML='';}}
  }
  function schedulePreview(){clearTimeout(previewTimer);previewRequest++;previewAbort?.abort();if($('ChallengeDialog').open)previewTimer=setTimeout(updatePreview,300);}
  $('ChallengeSetup').addEventListener('click',async()=>{
    $('ChallengeDialog').showModal();syncChallenge();
    if(!catalog){$('PreviewStatus').textContent='Loading LIBERO tasks…';await loadCatalog();}
    if(!$('ChallengeDialog').open)return;
    const suite=catalog?.suites?.find(s=>s.id===$('Suite').value);
    const previous=$('PreviewTask').value;
    $('PreviewTask').innerHTML=(suite?.tasks||[]).map(t=>`<option value="${t.id}">${esc(t.name)}</option>`).join('');
    $('PreviewTask').value=String(previous!==''&&selectedTasks().includes(Number(previous))?previous:(selectedTasks()[0]??0));
    if($('PreviewTask').value!=='')updatePreview();
  });
  for(const [id,mode] of [['ModeRandom','random'],['ModeFix','fix']])$(id).addEventListener('click',()=>{$('Mode').value=mode;syncChallenge();schedulePreview();});
  $('ChallengeClose').addEventListener('click',()=>$('ChallengeDialog').close());
  $('ChallengeDialog').addEventListener('click',e=>{if(e.target===$('ChallengeDialog'))$('ChallengeDialog').close();});
  $('ChallengeDialog').addEventListener('close',()=>{clearTimeout(previewTimer);previewRequest++;previewAbort?.abort();$('ChallengeSetup').focus();});
  $('ChallengeDialog').addEventListener('input',schedulePreview);
  $('PreviewTask').addEventListener('change',schedulePreview);
  $('ChallengeDialogEnabled').addEventListener('input',()=>{$('Challenge').checked=$('ChallengeDialogEnabled').checked;syncChallenge();});
  syncChallenge();
  root.liberoInit=function(){if(!catalog)loadCatalog();poll();};
})(typeof window !== 'undefined' ? window : globalThis);
