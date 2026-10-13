// One schema drives the challenge controls, saved config, and run payload.
const evalChallengeGroups = {
  camera: {title:'Camera Shift', id:'evalChCamEnabled', enabled:true, options:[
    ['perspective','Perspective','evalChCamPersp',true],
    ['affine','Affine','evalChCamAffine',true], ['rotation','Rotation','evalChCamRot',true]]},
  light: {title:'Light & Quality', id:'evalChLightEnabled', enabled:true, options:[
    ['brightness','Brightness','evalChLightBright',true], ['contrast','Contrast','evalChLightContrast',true],
    ['saturation','Saturation','evalChLightSat',true], ['color_jitter','Color Jitter','evalChLightJitter',true],
    ['shadow','Shadow','evalChLightShadow',false], ['noise','Noise','evalChLightNoise',false],
    ['blur','Blur','evalChLightBlur',false]]},
  occlusion: {title:'Occlusion', id:'evalChOcclusionEnabled', enabled:false, options:[]},
  robot: {title:'Robot Noise', id:'evalChRobotEnabled', enabled:false, options:[
    ['random_start','Random Start Trim','evalChRobotStart',true],
    ['initial_noise','Initial Noise','evalChRobotOffset',true],
    ['trajectory_jitter','Trajectory Jitter','evalChRobotJitter',true]]},
};

// group, key, label, min, max, step, default, technique
const evalChallengeFields = [
  ['camera','persp_mag','Perspective',0,.1,.005,.03,'perspective'],
  ['camera','tx','Translate X',-.08,.08,.005,.02,'affine'],
  ['camera','ty','Translate Y',-.08,.08,.005,.02,'affine'],
  ['camera','shear','Shear',-.06,.06,.005,0,'affine'],
  ['camera','scale','Scale',.9,1.1,.01,1,'affine'],
  ['camera','angle','Rotation (degrees)',-8,8,.5,2,'rotation'],
  ['light','brightness','Brightness',.5,1.5,.01,1.12,'brightness'],
  ['light','contrast','Contrast',.5,1.5,.01,1.12,'contrast'],
  ['light','saturation','Saturation',.3,1.7,.01,1.15,'saturation'],
  ['light','hue','Hue shift',-16,16,1,4,'color_jitter'],
  ['light','sat_off','Saturation offset',-30,30,1,0,'color_jitter'],
  ['light','val_off','Value offset',-30,30,1,0,'color_jitter'],
  ['light','shadow_a','Shadow strength',0,.8,.05,.25,'shadow'],
  ['light','noise_s','Noise sigma',0,24,.5,8,'noise'],
  ['light','blur_s','Blur sigma',0,2.4,.1,.8,'blur'],
  ['occlusion','x','Position X (%)',0,100,1,40,''],
  ['occlusion','y','Position Y (%)',0,100,1,40,''],
  ['occlusion','width','Width (%)',1,100,1,20,''],
  ['occlusion','height','Height (%)',1,100,1,20,''],
  ['robot','trim','Skip initial commands (frames)',0,30,1,3,'random_start'],
  ['robot','offset_sigma','Joint offset',0,3,.1,1,'initial_noise'],
  ['robot','jitter_s','Trajectory jitter sigma',0,2,.1,.5,'trajectory_jitter'],
];

function evalRenderChallenge() {
  const body = document.getElementById('evalChallengeBody');
  body.innerHTML = `
    <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin-bottom:10px">
      <label>Mode <select id="evalChallengeMode" class="form-input" style="width:110px" onchange="evalChallengeChanged()"><option value="random">Random</option><option value="fix">Fix</option></select></label>
      <label id="evalChallengeIntensityRow">Intensity (1–5) <input type="range" id="evalChallengeIntensity" min="1" max="5" value="3" oninput="evalChallengeChanged()"></label>
    </div>
    <div id="evalChallengeModeHelp" style="font-size:0.75rem;color:#888;margin-bottom:10px"></div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px">
    ${Object.entries(evalChallengeGroups).map(([group, spec]) => `
      <div style="background:#f9f9f9;border:1px solid #e0e0e0;border-radius:6px;padding:10px">
        <label style="font-weight:600;font-size:0.82rem"><input type="checkbox" id="${spec.id}" ${spec.enabled?'checked':''} onchange="evalChallengeChanged()"> ${spec.title}</label>
        ${spec.options.map(([,label,id,checked]) => `<div style="font-size:0.8rem;padding-left:18px;margin-top:4px"><label><input type="checkbox" id="${id}" ${checked?'checked':''} onchange="evalChallengeChanged()"> ${label}</label></div>`).join('')}
        ${group==='robot'?'<p style="font-size:0.72rem;color:#888">Trim skips initial commands. Joint noise affects observations and outgoing arm commands; the gripper stays unchanged.</p>':''}
        ${group==='occlusion'?'<p style="font-size:0.72rem;color:#888">Black rectangle with a stable position and size throughout each episode.</p>':''}
        ${evalChallengeFields.filter(f=>f[0]===group).map(([g,key,label,min,max,step,value,tech])=>`
          <label data-challenge-field="${g}" data-technique="${tech}" style="font-size:0.75rem;display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:6px">${label}
            <input id="evalFix_${g}_${key}" aria-label="${spec.title}: ${label}" type="number" min="${min}" max="${max}" step="${step}" value="${value}" style="width:85px;padding:4px" onchange="evalChallengeChanged()">
          </label>`).join('')}
        ${group==='light'?'<label data-challenge-field="light" data-technique="shadow" style="font-size:0.75rem;display:flex;justify-content:space-between;margin-top:6px">Shadow direction <select id="evalFix_light_shadow_dir" onchange="evalChallengeChanged()"><option>left</option><option>right</option><option>top</option><option>bottom</option></select></label>':''}
      </div>`).join('')}
    </div>
    <section style="margin-top:16px;border-top:1px solid #ddd;padding-top:12px">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px">
        <strong style="font-size:0.85rem;color:#6a1b9a">Preview (up to 5 episodes)</strong>
        <label style="font-size:0.78rem">Source dataset <select id="evalChallengePreviewDataset" class="form-input" style="width:200px" onchange="evalScheduleChallengePreview()"><option value="">Select dataset</option></select></label>
        <label style="font-size:0.78rem">Challenge episode <input id="evalChallengePreviewEpisode" type="number" min="1" value="1" style="width:65px;padding:4px" onchange="evalScheduleChallengePreview()"></label>
        <button type="button" class="btn btn-sm btn-ghost" onclick="evalLoadChallengePreviewSources()">Refresh preview</button>
      </div>
      <div style="font-size:0.72rem;color:#888;margin-bottom:8px">Dataset images show the camera effects for successive challenge episodes. Robot settings are listed below each pair; preview does not move the robot.</div>
      <div id="evalChallengePreviewStatus" role="status" style="font-size:0.78rem;color:#888;margin-bottom:8px">Select a dataset to preview.</div>
      <div id="evalChallengePreviewGrid" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px"></div>
    </section>`;
  evalSyncChallengeControls();
}

function evalSyncChallengeControls() {
  const fix = document.getElementById('evalChallengeMode').value === 'fix';
  document.getElementById('evalChallengeIntensityRow').style.display = fix ? 'none' : '';
  document.getElementById('evalChallengeModeHelp').textContent = fix
    ? 'Fix uses the values below. Noise values set the amount of noise; the episode number keeps comparisons repeatable.'
    : 'Random samples values at the selected intensity. The same episode number uses the same challenge across models.';
  document.querySelectorAll('[data-challenge-field]').forEach(el => {
    const spec = evalChallengeGroups[el.dataset.challengeField];
    const option = spec.options.find(o => o[0] === el.dataset.technique);
    const enabled = document.getElementById(spec.id).checked && (!option || document.getElementById(option[2]).checked);
    el.style.display = fix && enabled ? 'flex' : 'none';
  });
}

function evalChallengeChanged() {
  evalSyncChallengeControls();
  evalConfigChanged();
}

function evalGetChallengeConfig() {
  const config = {
    enabled: document.getElementById('evalChallengeEnabled').checked,
    mode: document.getElementById('evalChallengeMode').value || 'random',
    intensity: Number(document.getElementById('evalChallengeIntensity').value) || 3,
  };
  for (const [group,spec] of Object.entries(evalChallengeGroups)) {
    config[group] = {enabled: document.getElementById(spec.id).checked};
    for (const [key,,id] of spec.options) config[group][key] = document.getElementById(id).checked;
  }
  const fixed = {camera:{},light:{},robot:{}};
  const occ = {};
  for (const [group,key,,min,max,,fallback] of evalChallengeFields) {
    const raw = document.getElementById(`evalFix_${group}_${key}`).value;
    const number = raw === '' ? fallback : Number(raw);
    const value = Number.isFinite(number) ? Math.max(min,Math.min(max,number)) : fallback;
    if (group === 'occlusion') occ[key] = value / 100;
    else fixed[group][key] = value;
  }
  fixed.light.shadow_dir = document.getElementById('evalFix_light_shadow_dir').value || 'left';
  fixed.light.occlusion = [occ.x,occ.y,occ.width,occ.height];
  config.fixed_params = fixed;
  return config;
}

function evalSetChallengeConfig(config = {}) {
  document.getElementById('evalChallengeEnabled').checked = !!config.enabled;
  document.getElementById('evalChallengeMode').value = config.mode === 'fix' ? 'fix' : 'random';
  document.getElementById('evalChallengeIntensity').value = config.intensity ?? 3;
  for (const [group,spec] of Object.entries(evalChallengeGroups)) {
    const values = config[group] || {};
    document.getElementById(spec.id).checked = values.enabled ?? spec.enabled;
    for (const [key,,id,fallback] of spec.options) document.getElementById(id).checked = values[key] ?? fallback;
  }
  const fixed = config.fixed_params || {};
  const occ = fixed.light?.occlusion;
  const occValues = occ ? Object.fromEntries(['x','y','width','height'].map((k,i)=>[k,occ[i]*100])) : {};
  for (const [group,key,,,,,fallback] of evalChallengeFields) {
    document.getElementById(`evalFix_${group}_${key}`).value = (group === 'occlusion' ? occValues[key] : fixed[group]?.[key]) ?? fallback;
  }
  document.getElementById('evalFix_light_shadow_dir').value = fixed.light?.shadow_dir || 'left';
  evalSyncChallengeControls();
  evalScheduleChallengePreview();
}

let evalChallengePreviewTimer = null;
let evalChallengePreviewRequest = 0;
let evalChallengePreviewController = null;

function evalChallengeEscape(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

async function evalLoadChallengePreviewSources() {
  const select = document.getElementById('evalChallengePreviewDataset');
  const status = document.getElementById('evalChallengePreviewStatus');
  const previous = select.value;
  try {
    const response = await fetch('/api/datasets');
    if (!response.ok) throw new Error(`Unable to load datasets (HTTP ${response.status})`);
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'Unable to load datasets');
    const datasets = data.datasets || [];
    select.innerHTML = '<option value="">Select dataset</option>' + datasets.map(d => `<option value="${evalChallengeEscape(d.name)}">${evalChallengeEscape(d.name)}</option>`).join('');
    select.value = datasets.some(d => d.name === previous) ? previous : (datasets[0]?.name || '');
    if (!select.value) {
      await evalUpdateChallengePreview();
      status.textContent = 'No datasets available. Add a dataset with camera videos to preview.';
      return;
    }
    if (!previous && typeof evalSession !== 'undefined' && evalSession) {
      document.getElementById('evalChallengePreviewEpisode').value = (evalSession.completed || 0) + 1;
    }
    await evalUpdateChallengePreview();
  } catch(error) {
    status.textContent = error.message;
  }
}

function evalScheduleChallengePreview() {
  clearTimeout(evalChallengePreviewTimer);
  // Invalidate in-flight responses as soon as a control changes.
  evalChallengePreviewRequest++;
  if (evalChallengePreviewController) evalChallengePreviewController.abort();
  if (!document.getElementById('evalChallengePreviewDataset')?.value) {
    document.getElementById('evalChallengePreviewGrid').innerHTML = '';
    document.getElementById('evalChallengePreviewStatus').textContent = 'Select a dataset to preview.';
    return;
  }
  if (document.getElementById('evalChallengeBody').style.display === 'none') return;
  evalChallengePreviewTimer = setTimeout(evalUpdateChallengePreview, 300);
}

async function evalUpdateChallengePreview() {
  const dataset = document.getElementById('evalChallengePreviewDataset').value;
  const grid = document.getElementById('evalChallengePreviewGrid');
  const status = document.getElementById('evalChallengePreviewStatus');
  const request = ++evalChallengePreviewRequest;
  if (evalChallengePreviewController) evalChallengePreviewController.abort();
  if (!dataset) { grid.innerHTML = ''; status.textContent = 'Select a dataset to preview.'; return; }
  evalChallengePreviewController = new AbortController();
  status.textContent = 'Generating preview...';
  grid.style.opacity = '0.5';
  try {
    const response = await fetch('/api/eval/challenge-preview', {
      method:'POST', headers:{'Content-Type':'application/json'}, signal:evalChallengePreviewController.signal,
      body:JSON.stringify({dataset, count:5,
        episode:Math.max(1, Number(document.getElementById('evalChallengePreviewEpisode').value) || 1),
        fps:Number(document.getElementById('evalFps').value) || 30,
        duration:Number(document.getElementById('evalDuration').value) || 12,
        width:Number(document.getElementById('evalWidth').value) || 320,
        height:Number(document.getElementById('evalHeight').value) || 240,
        cam_count:document.getElementById('evalCamCount').value || 'all',
        challenge_env:evalGetChallengeConfig()}),
    });
    if (!response.ok) throw new Error(`Preview failed (HTTP ${response.status}). Please refresh and try again.`);
    const data = await response.json();
    if (request !== evalChallengePreviewRequest) return;
    if (!data.ok) throw new Error(data.error || 'Unable to generate preview');
    grid.innerHTML = (data.previews || []).map(p => {
      const robot = p.params?.robot;
      const robotInfo = robot ? [
        `Trim: ${robot.trim || 0} frames`,
        `Joint offset: ${(robot.joint_offset || []).map(v => Number(v).toFixed(2)).join(', ') || 'off'}`,
        `Jitter sigma: ${Number(robot.jitter_s || 0).toFixed(2)}`,
      ].join(' · ') : 'Robot Noise: off';
      return `<div style="border:1px solid #e0e0e0;border-radius:6px;padding:8px">
        <div style="font-size:0.73rem;color:#666;margin-bottom:5px">Source ep ${evalChallengeEscape(p.source_episode)} · ${evalChallengeEscape(p.camera)} · Challenge ep ${evalChallengeEscape(p.challenge_episode)}</div>
        <div style="display:flex;gap:8px">
          <div style="flex:1;min-width:0"><img alt="Original" src="data:image/jpeg;base64,${p.original}" style="width:100%;border-radius:4px"><div style="font-size:0.7rem;color:#888">Original</div></div>
          <div style="flex:1;min-width:0"><img alt="Challenge preview" src="data:image/jpeg;base64,${p.augmented}" style="width:100%;border-radius:4px"><div style="font-size:0.7rem;color:#6a1b9a">Challenge</div></div>
        </div><div style="font-size:0.7rem;color:#666;margin-top:6px">${evalChallengeEscape(robotInfo)}</div>
      </div>`;
    }).join('');
    status.textContent = data.previews?.length ? `${data.previews.length} camera image pairs · ${evalGetChallengeConfig().enabled ? 'Challenge enabled' : 'Challenge disabled — original images'}` : 'No preview images available.';
  } catch(error) {
    if (request !== evalChallengePreviewRequest || error.name === 'AbortError') return;
    status.textContent = error.message;
    grid.innerHTML = '';
  } finally {
    if (request === evalChallengePreviewRequest) grid.style.opacity = '1';
  }
}

evalRenderChallenge();
