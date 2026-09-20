const state={tab:'audio',file:null};
const $=selector=>document.querySelector(selector);
const fileInput=$('#fileInput');
const dropzone=$('#dropzone');
const analyzeButton=$('#analyzeBtn');
const audioPreview=$('#audioPreview');
const imagePreview=$('#imagePreview');

function setStatus(text,error=false){const element=$('#status');element.textContent=text;element.className=`status${error?' error':''}`}
function configureTab(tab){
  state.tab=tab;
  document.querySelectorAll('.tab').forEach(button=>button.classList.toggle('active',button.dataset.tab===tab));
  const audio=tab==='audio';
  $('#uploadTitle').textContent=audio?'Upload a bird recording':'Upload a bird image';
  $('#uploadHint').textContent=audio?'WAV, MP3, OGG, FLAC, or M4A. The four audio models use the same 40-feature MFCC representation.':'JPG, PNG, or WEBP. Every vision model receives a normalized 224 × 224 crop.';
  fileInput.accept=audio?'audio/wav,audio/mpeg,audio/ogg,audio/flac,audio/mp4':'image/jpeg,image/png,image/webp';
  $('#dropIcon').textContent=audio?'↥':'◈';
  clearFile();
}
function clearFile(){state.file=null;fileInput.value='';analyzeButton.disabled=true;audioPreview.hidden=true;imagePreview.hidden=true;audioPreview.removeAttribute('src');imagePreview.removeAttribute('src');$('#dropText').textContent='Drop a file here or browse';$('#fileMeta').textContent='Maximum recommended size: 50 MB'}
function selectFile(file){
  if(!file)return;
  state.file=file;analyzeButton.disabled=false;$('#dropText').textContent=file.name;$('#fileMeta').textContent=`${(file.size/1024/1024).toFixed(2)} MB`;
  const url=URL.createObjectURL(file);
  if(state.tab==='audio'){audioPreview.src=url;audioPreview.hidden=false;imagePreview.hidden=true}else{imagePreview.src=url;imagePreview.hidden=false;audioPreview.hidden=true}
  setStatus('Sample ready. Start analysis when you are ready.');
}
function renderResults(data){
  $('#results').hidden=false;
  const consensus=data.consensus;
  $('#summaryTitle').textContent=consensus.agreement;
  $('#summaryText').textContent=`The ensemble decision is based on ${data.results.length} independently evaluated models.`;
  $('#consensusSpecies').textContent=consensus.species;
  $('#resultMeta').textContent=data.modality==='image'?`Device: ${data.device}`:`Sample rate: ${data.sample_rate} Hz`;
  $('#modelGrid').innerHTML=data.results.map(result=>`<article class="model-card"><h3>${result.model}</h3><span class="label">PREDICTED SPECIES</span><div class="species">${result.species}</div><div class="confidence-row"><span>Confidence</span><strong>${result.confidence.toFixed(2)}%</strong></div><div class="gauge"><i style="width:${Math.min(100,result.confidence)}%"></i></div><div class="label top-title">TOP 3 PREDICTIONS</div>${result.top_three.map(item=>`<div class="top-row"><span>${item.species}</span><span>${item.probability.toFixed(2)}%</span></div>`).join('')}<div class="latency"><span>Inference latency</span><strong>${result.latency_ms.toFixed(2)} ms</strong></div></article>`).join('');
  $('#results').scrollIntoView({behavior:'smooth',block:'start'});
}
async function analyze(){
  if(!state.file)return;
  analyzeButton.disabled=true;analyzeButton.innerHTML='Analyzing <span>…</span>';setStatus('Running all four models independently…');
  const form=new FormData();form.append('file',state.file);
  try{const response=await fetch(`/predict/${state.tab}`,{method:'POST',body:form});const data=await response.json();if(!response.ok)throw new Error(data.error||'Inference failed');renderResults(data);setStatus('Analysis complete.');}
  catch(error){setStatus(error.message,true)}
  finally{analyzeButton.disabled=false;analyzeButton.innerHTML='Analyze sample <span>→</span>'}
}

document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>configureTab(button.dataset.tab)));
dropzone.addEventListener('click',()=>fileInput.click());fileInput.addEventListener('change',event=>selectFile(event.target.files[0]));
['dragenter','dragover'].forEach(type=>dropzone.addEventListener(type,event=>{event.preventDefault();dropzone.classList.add('dragging')}));['dragleave','drop'].forEach(type=>dropzone.addEventListener(type,event=>{event.preventDefault();dropzone.classList.remove('dragging')}));dropzone.addEventListener('drop',event=>selectFile(event.dataTransfer.files[0]));analyzeButton.addEventListener('click',analyze);
