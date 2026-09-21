/* video.js — client logic for the /video analysis page */

const state = { file: null };
const $ = selector => document.querySelector(selector);

const fileInput     = $('#fileInput');
const dropzone      = $('#dropzone');
const analyzeBtn    = $('#analyzeBtn');
const videoPreview  = $('#videoPreview');
const progressWrap  = $('#progressWrap');
const progressLabel = $('#progressLabel');

/* ── status bar ─────────────────────────────────────────────────────────── */
function setStatus(text, error = false) {
  const el = $('#status');
  el.textContent = text;
  el.className = `status${error ? ' error' : ''}`;
}

/* ── file selection ──────────────────────────────────────────────────────── */
function clearFile() {
  state.file = null;
  fileInput.value = '';
  analyzeBtn.disabled = true;
  videoPreview.hidden = true;
  videoPreview.removeAttribute('src');
  $('#dropText').textContent = 'Drop a video here or browse';
  $('#fileMeta').textContent = 'Maximum recommended size: 200 MB';
  $('#results').hidden = true;
  progressWrap.hidden = true;
}

function selectFile(file) {
  if (!file) return;
  state.file = file;
  analyzeBtn.disabled = false;
  $('#dropText').textContent = file.name;
  $('#fileMeta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  videoPreview.hidden = false;
  setStatus('Video ready. Start analysis when you are ready.');
}

/* ── render helpers ──────────────────────────────────────────────────────── */
function buildModelCard(result) {
  return `<article class="model-card">
    <h3>${result.model}</h3>
    <span class="label">PREDICTED SPECIES</span>
    <div class="species">${result.species}</div>
    <div class="confidence-row"><span>Confidence</span><strong>${result.confidence.toFixed(2)}%</strong></div>
    <div class="gauge"><i style="width:${Math.min(100, result.confidence)}%"></i></div>
    <div class="label top-title">TOP 3 PREDICTIONS</div>
    ${result.top_three.map(item => `<div class="top-row"><span>${item.species}</span><span>${item.probability.toFixed(2)}%</span></div>`).join('')}
    <div class="latency"><span>Avg inference latency</span><strong>${result.latency_ms.toFixed(2)} ms</strong></div>
  </article>`;
}

function renderResults(data) {
  /* multimodal fusion banner */
  const mm = data.multimodal;
  $('#fusionAgreement').textContent = mm.agreement;
  $('#fusionSpecies').textContent   = mm.species;
  $('#fusionText').textContent      = `Fused from audio consensus and visual consensus across ${data.frame_count} sampled frame${data.frame_count !== 1 ? 's' : ''} (${data.duration_s}s @ ${data.fps} fps).`;

  /* audio section */
  const audio = data.audio;
  if (audio && audio.results && audio.results.length) {
    const ac = audio.consensus;
    $('#audioMeta').textContent          = `Sample rate: ${audio.sample_rate} Hz`;
    $('#audioConsensusTitle').textContent = ac.agreement;
    $('#audioConsensusText').textContent  = `${audio.results.length} audio models evaluated.`;
    $('#audioConsensusSpecies').textContent = ac.species;
    $('#audioGrid').innerHTML = audio.results.map(buildModelCard).join('');
  } else {
    const err = audio && audio.error ? audio.error : 'No audio results.';
    $('#audioConsensusTitle').textContent = 'Audio unavailable';
    $('#audioConsensusText').textContent  = err;
    $('#audioConsensusSpecies').textContent = '—';
    $('#audioGrid').innerHTML = `<p class="status error">${err}</p>`;
  }

  /* image section */
  const image = data.image;
  if (image && image.results && image.results.length) {
    const ic = image.consensus;
    $('#imageMeta').textContent          = `Device: ${image.device}`;
    $('#imageConsensusTitle').textContent = ic.agreement;
    $('#imageConsensusText').textContent  = `${image.results.length} vision models evaluated across ${data.frame_count} frame${data.frame_count !== 1 ? 's' : ''}.`;
    $('#imageConsensusSpecies').textContent = ic.species;
    $('#frameBadge').textContent = `(${data.frame_count} frames)`;
    $('#imageGrid').innerHTML = image.results.map(buildModelCard).join('');
  } else {
    $('#imageConsensusTitle').textContent = 'Vision unavailable';
    $('#imageConsensusSpecies').textContent = '—';
    $('#imageGrid').innerHTML = '<p class="status error">No image results.</p>';
  }

  $('#results').hidden = false;
  $('#results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── analysis ────────────────────────────────────────────────────────────── */
const steps = [
  'Uploading video…',
  'Extracting audio track…',
  'Sampling frames…',
  'Running audio models…',
  'Running vision models…',
  'Fusing multimodal results…',
];

async function analyze() {
  if (!state.file) return;

  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = 'Analysing <span>…</span>';
  $('#results').hidden = true;
  progressWrap.hidden = false;

  /* cycle through progress labels while waiting */
  let stepIdx = 0;
  progressLabel.textContent = steps[0];
  const labelTimer = setInterval(() => {
    stepIdx = (stepIdx + 1) % steps.length;
    progressLabel.textContent = steps[stepIdx];
  }, 2000);

  const form = new FormData();
  form.append('file', state.file);
  form.append('n_frames', $('#nFrames').value);

  try {
    setStatus('Running multimodal analysis — this may take a moment…');
    const response = await fetch('/predict/video', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Inference failed');
    renderResults(data);
    setStatus('Multimodal analysis complete.');
  } catch (err) {
    setStatus(err.message, true);
    $('#results').hidden = true;
  } finally {
    clearInterval(labelTimer);
    progressWrap.hidden = true;
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = 'Analyze video <span>→</span>';
  }
}

/* ── event wiring ────────────────────────────────────────────────────────── */
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', event => selectFile(event.target.files[0]));

['dragenter', 'dragover'].forEach(type =>
  dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add('dragging'); })
);
['dragleave', 'drop'].forEach(type =>
  dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove('dragging'); })
);
dropzone.addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));

analyzeBtn.addEventListener('click', analyze);
