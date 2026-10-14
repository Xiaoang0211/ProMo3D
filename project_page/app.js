const input = document.querySelector('#input-video');
const prediction = document.querySelector('#prediction-video');
const play = document.querySelector('#play-button');
const seek = document.querySelector('#seek');
const counter = document.querySelector('#frame-counter');
let samples, currentSample, currentView = 'coarse', playing = false, loadId = 0;
const viewNames = {scores: 'PMD dynamic scores', coarse: 'PMD coarse', refined: 'PMD + SAM2'};
const mediaCache = new Map();

function mediaSource(path) {
  // Blob URLs support seeking even on hosts that do not serve byte ranges.
  if (!mediaCache.has(path)) mediaCache.set(path, fetch(path).then(response => {
    if (!response.ok) throw Error(`Unable to load ${path}`);
    return response.blob();
  }).then(blob => URL.createObjectURL(blob)));
  return mediaCache.get(path);
}

function selected(selector, value, key) {
  document.querySelectorAll(selector).forEach(button => button.setAttribute('aria-pressed', String(button.dataset[key] === value)));
}
function playbackState(value) {
  playing = value;
  play.textContent = value ? 'Pause' : 'Play';
  play.setAttribute('aria-label', value ? 'Pause both videos' : 'Play both videos');
}
function stop() { input.pause(); prediction.pause(); playbackState(false); }
async function start() {
  try { await Promise.all([input.play(), prediction.play()]); playbackState(true); }
  catch { stop(); }
}
function metadata(video) {
  return new Promise((resolve, reject) => {
    video.addEventListener('loadedmetadata', resolve, {once:true});
    video.addEventListener('error', reject, {once:true});
    video.load();
  });
}
function seekVideo(video, time) {
  if (Math.abs(video.currentTime - time) < .01 && !video.seeking) return Promise.resolve();
  return new Promise(resolve => {
    video.addEventListener('seeked', resolve, {once:true});
    video.currentTime = time;
  });
}
async function showSample(id, view, preserveTime = false) {
  if (!samples) return;
  const revision = ++loadId;
  const resume = playing, time = preserveTime ? input.currentTime : 0;
  const sameSequence = currentSample?.id === id;
  stop();
  currentSample = samples.find(sample => sample.id === id);
  currentView = view;
  selected('[data-sequence]', id, 'sequence');
  selected('[data-view]', view, 'view');
  document.querySelector('#sequence-description').textContent = currentSample.description;
  document.querySelector('#prediction-label').textContent = viewNames[view];
  play.disabled = true;
  seek.disabled = true;
  document.querySelector('.video-pair').setAttribute('aria-busy', 'true');
  const sample = currentSample;
  try {
    const [inputUrl, predictionUrl] = await Promise.all([
      sameSequence ? null : mediaSource(sample.views.input.video), mediaSource(sample.views[view].video)
    ]);
    if (revision !== loadId) return;
    const loading = [];
    if (!sameSequence) {
      input.src = inputUrl;
      input.dataset.asset = sample.views.input.video;
      input.poster = sample.views.input.poster;
      loading.push(metadata(input));
    }
    prediction.src = predictionUrl;
    prediction.dataset.asset = sample.views[view].video;
    prediction.poster = sample.views[view].poster;
    loading.push(metadata(prediction));
    await Promise.all(loading);
    if (revision !== loadId) return;
    const targetTime = Math.min(time, input.duration - .01);
    await Promise.all([seekVideo(input, targetTime), seekVideo(prediction, targetTime)]);
    if (revision !== loadId) return;
    document.querySelector('.video-pair').style.setProperty('--video-ratio', input.videoWidth / input.videoHeight);
    updatePosition();
    play.disabled = seek.disabled = false;
    document.querySelector('.video-pair').setAttribute('aria-busy', 'false');
    if (resume) await start();
  } catch {
    if (revision !== loadId) return;
    document.querySelector('#gallery-error').hidden = false;
    document.querySelector('#gallery-error').textContent = 'The sequence could not be loaded. Please reload the page.';
  }
}
function updatePosition() {
  if (!currentSample || !Number.isFinite(input.duration)) return;
  seek.value = Math.round(1000 * input.currentTime / input.duration);
  const frame = Math.min(currentSample.frames - 1, Math.floor(input.currentTime * currentSample.fps));
  counter.textContent = `Frame ${frame + 1} / ${currentSample.frames}`;
  if (playing && Math.abs(input.currentTime - prediction.currentTime) > .12) prediction.currentTime = input.currentTime;
}
play.addEventListener('click', () => playing ? stop() : start());
seek.addEventListener('input', () => { input.currentTime = prediction.currentTime = input.duration * seek.value / 1000; updatePosition(); });
document.querySelector('#speed').addEventListener('change', event => { input.playbackRate = prediction.playbackRate = Number(event.target.value); });
input.addEventListener('timeupdate', updatePosition);
input.addEventListener('ended', stop);
document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
document.querySelectorAll('[data-sequence]').forEach(button => button.addEventListener('click', () => showSample(button.dataset.sequence, currentView)));
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => { if (currentSample) showSample(currentSample.id, button.dataset.view, true); }));
fetch('assets/samples.json').then(response => { if (!response.ok) throw Error('Missing sample manifest'); return response.json(); }).then(data => { samples = data; showSample('rhino', 'coarse'); }).catch(() => { counter.textContent = 'Sequence unavailable'; play.disabled = true; });

const panels = {
  layerwise: ['layerwise.png', 'Layer-wise motion decodability on Pi3X, DA3, and DUSt3R.'],
  pca: ['pca.png', 'Pi3X: PCA of the probe-induced representation, with low-rank and spectral analyses.'],
  starvation: ['starvation.png', 'Token-level normalized gain as training scene diversity decreases from 160 to 10 videos.']
};
document.querySelectorAll('[data-panel]').forEach(button => button.addEventListener('click', () => {
  selected('[data-panel]', button.dataset.panel, 'panel');
  const [file, caption] = panels[button.dataset.panel];
  const img = document.querySelector('#evidence-image');
  img.src = `assets/figures/${file}`; img.alt = caption;
  document.querySelector('#evidence-caption').textContent = `${caption} Click the figure to enlarge.`;
  document.querySelector('#pca-note').hidden = button.dataset.panel !== 'pca';
}));
const dialog = document.querySelector('#figure-dialog');
document.querySelector('#open-figure').addEventListener('click', () => {
  const source = document.querySelector('#evidence-image');
  document.querySelector('#large-figure').src = source.src;
  document.querySelector('#large-figure').alt = source.alt;
  document.querySelector('#large-caption').textContent = source.alt;
  dialog.showModal();
});
document.querySelector('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
async function copy(button, selector) {
  const label = button.textContent;
  try { await navigator.clipboard.writeText(document.querySelector(selector).textContent.trim()); button.textContent = 'Copied'; }
  catch { button.textContent = 'Select the text below'; }
  setTimeout(() => { button.textContent = label; }, 1800);
}
document.querySelector('#copy-commands').addEventListener('click', event => copy(event.currentTarget, '#commands'));
document.querySelector('#copy-citation').addEventListener('click', event => copy(event.currentTarget, '#bibtex'));
