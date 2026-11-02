// Welke bak? Sorteerspel en explainer.
//
// Alle cijfers komen uit bestanden die de Python-scripts schreven (data/web.json en
// ../results/*.json); voorspellingen komen live uit de ONNX-modellen in ../model/.
// Lukt het laden van ONNX Runtime niet, dan gebruikt de pagina de vooraf in Python
// berekende kansen voor de voorbeeldfoto's en schakelt uploads en schuifjes uit.

import { preprocess, toRGB, brightness, boxBlur, MEAN, STD, DARK, REFERENCE_WIDTH } from './preprocess.js';

const ORT_VERSION = '1.20.1';
const MAX_UPLOAD_BYTES = 20e6;
const MAX_SIDE = 1024; // grote uploads eerst verkleinen, anders wordt het schalen traag
const ROUNDS = 8;
const SURE = 0.8;

const MODELS = [
  { id: 'cnn', name: 'Eigen CNN', short: 'CNN' },
  { id: 'mobilenet', name: 'MobileNet', short: 'MobileNet' },
];

const SLIDERS = [
  { id: 'licht', label: 'Licht', min: 0.25, max: 2, step: 0.05, neutral: 1, format: (v) => `${fmtNum(v, 2)}×` },
  { id: 'draaien', label: 'Draaien', min: 0, max: 180, step: 5, neutral: 0, format: (v) => `${v}°` },
  { id: 'vervagen', label: 'Vervagen', min: 0, max: 16, step: 1, neutral: 0, format: (v) => (v ? `straal ${v}` : 'scherp') },
  // Hoeveel donkere tafel om de foto heen: de foto wordt (1 - waarde) zo groot
  { id: 'tafel', label: 'Donkere tafel', min: 0, max: 0.65, step: 0.05, neutral: 0, format: (v) => (v > 0 ? `foto ${Math.round((1 - v) * 100)}%` : 'uit') },
];

const state = {
  data: null,
  history: {},
  evaluation: {},
  variations: null,
  onnx: null,
  sessions: null,
  liveProbs: new Map(),
  source: null,
  base: null,
  params: Object.fromEntries(SLIDERS.map((s) => [s.id, s.neutral])),
  game: null,
  curveMetric: 'accuracy',
  testBusy: false,
  testPending: false,
};

// ---------- hulpjes ----------

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, html = '') => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  node.innerHTML = html;
  return node;
};
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function fmtNum(x, digits = 0) {
  return x.toLocaleString('nl-NL', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
// Nooit 100% tonen als het model niet precies 1 gaf; toPrecision haalt float-ruis weg (0.795 * 100)
function fmtPct(x, digits = 0) {
  const factor = 10 ** digits;
  let v = Math.round(Number((x * 100).toPrecision(12)) * factor) / factor;
  if (v >= 100 && x < 1) v = 100 - 1 / factor;
  return `${fmtNum(v, digits)}%`;
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const argmax = (arr) => arr.reduce((best, v, i) => (v > arr[best] ? i : best), 0);
const sum = (arr) => arr.reduce((a, b) => a + b, 0);
const cls = (k) => state.data.classes[k];
const binColor = (k) => `var(--bin-${k})`;
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('kon afbeelding niet openen'));
    img.src = url;
  });
}

function canvasOf(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function drawCanvas(target, source) {
  target.width = source.width;
  target.height = source.height;
  target.getContext('2d').drawImage(source, 0, 0);
}

function setStatus(stateName, text) {
  const s = $('#status');
  s.dataset.state = stateName;
  s.textContent = text;
}

const exampleUrl = (ex) => `../assets/examples/${ex.file}`;

// ---------- tooltip ----------

function showTip(html, x, y) {
  const t = $('#tooltip');
  t.innerHTML = html;
  t.hidden = false;
  const pad = 12;
  const { width, height } = t.getBoundingClientRect();
  let left = x + pad;
  let top = y + pad;
  if (left + width > window.innerWidth - 8) left = x - width - pad;
  if (top + height > window.innerHeight - 8) top = y - height - pad;
  t.style.left = `${Math.max(8, left)}px`;
  t.style.top = `${Math.max(8, top)}px`;
}
const hideTip = () => { $('#tooltip').hidden = true; };
document.addEventListener('pointermove', (e) => {
  const target = e.target.closest?.('[data-tip]');
  if (target) showTip(target.dataset.tip, e.clientX, e.clientY);
  else if (!e.target.closest?.('[data-crosshair]')) hideTip();
});

// ---------- gedeelde onderdelen ----------

function segmented(container, options, current, onChange) {
  container.innerHTML = '';
  for (const opt of options) {
    const b = el('button', { type: 'button', 'aria-pressed': String(opt.id === current) }, esc(opt.label));
    b.addEventListener('click', () => {
      for (const other of container.children) other.setAttribute('aria-pressed', 'false');
      b.setAttribute('aria-pressed', 'true');
      onChange(opt.id);
    });
    container.append(b);
  }
}

const chip = (model, p) => `<span class="chip ${model.id}"><i></i>${esc(model.short)} ${fmtPct(p)}</span>`;

// Vijf bakken. marks[k] = { chips: [html], truth, dim }; interactive maakt er knoppen van
function bins({ interactive = false, marks = {}, small = false } = {}) {
  return `<div class="bins">${state.data.classes.map((c, k) => {
    const m = marks[k] || {};
    const tag = interactive ? 'button' : 'div';
    const attrs = interactive ? ` type="button" data-pick="${k}" aria-label="${esc(cap(c))}"` : '';
    return `<${tag} class="bin${m.truth ? ' truth' : ''}${m.dim ? ' dim' : ''}" data-k="${k}"${attrs}>
      <span class="chips">${(m.chips || []).join('')}</span>
      <span class="bin-lid"></span>
      <span class="bin-body"${small ? ' style="height:54px"' : ''}><span class="bin-label">${esc(cap(c))}</span></span>
    </${tag}>`;
  }).join('')}</div>`;
}

function vbars(probs) {
  const k = argmax(probs);
  return `<div class="vbars">${state.data.classes.map((c, i) => `
    <div class="vbar${i === k ? ' win' : ''}" style="--c:${binColor(i)}">
      <div class="vbar-top"><span>${esc(cap(c))}</span><span class="mono">${fmtPct(probs[i])}</span></div>
      <div class="vbar-track"><div class="vbar-fill" style="width:${(probs[i] * 100).toFixed(1)}%"></div></div>
    </div>`).join('')}</div>`;
}

function table(headers, rows) {
  return `<div class="table-scroll"><table class="data"><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function niceTicks(max, count = 4) {
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

// Modellijn + markering, zonder kleur (zoals een bouwtekening):
// eigen CNN = doorgetrokken lijn met dicht vierkant, MobileNet = stippellijn met open rondje
const LINE = {
  cnn: { color: 'var(--ink)', width: 2, dash: '' },
  mobilenet: { color: 'var(--ink)', width: 2, dash: '6 4' },
};
function marker(id, x, y, r = 5) {
  return id === 'cnn'
    ? `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" fill="var(--ink)"/>`
    : `<circle cx="${x}" cy="${y}" r="${r - 0.5}" fill="var(--bg)" stroke="var(--ink)" stroke-width="2"/>`;
}
const linePath = (id, d) => `<path class="series-line" d="${d}" stroke="${LINE[id].color}" stroke-width="${LINE[id].width}"${LINE[id].dash ? ` stroke-dasharray="${LINE[id].dash}"` : ''}/>`;
function modelLegend(container) {
  container.innerHTML = MODELS.map((m) => `<span class="key"><svg width="34" height="12">
    ${linePath(m.id, 'M0,6L34,6')}${marker(m.id, 17, 6, 4)}</svg>${esc(m.name)}</span>`).join('');
}

// ---------- voorspellen ----------

// Spel, lab en controle kunnen tegelijk willen rekenen; ONNX Runtime Web wil één run tegelijk per sessie
let runQueue = Promise.resolve();
function runModel(id, tensor, size) {
  const job = runQueue.then(() => state.sessions[id].run({ image: new ort.Tensor('float32', tensor, [1, 3, size, size]) }));
  runQueue = job.catch(() => {});
  return job;
}

async function predict(canvas, example = null) {
  const { width: w, height: h } = canvas;
  const rgb = toRGB(canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data, w, h);
  const result = { probs: {}, crops: {}, tensors: {}, live: !!state.sessions };
  for (const m of MODELS) {
    const size = state.data.image_size[m.id];
    const { crop, tensor } = preprocess(rgb, w, h, size);
    result.crops[m.id] = crop;
    result.tensors[m.id] = tensor;
    if (state.sessions) {
      const out = await runModel(m.id, tensor, size);
      result.probs[m.id] = Array.from(out.probabilities.data);
    } else if (example) {
      result.probs[m.id] = example.probabilities[m.id];
    }
  }
  return result;
}

async function exampleCanvas(ex) {
  const img = await loadImage(exampleUrl(ex));
  const c = canvasOf(img.naturalWidth, img.naturalHeight);
  c.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0);
  return c;
}

// Live kansen voor een voorbeeldfoto (één keer berekend), anders de Python-referentie
async function exampleProbs(ex) {
  if (!state.sessions) return ex.probabilities;
  if (!state.liveProbs.has(ex.file)) state.liveProbs.set(ex.file, (await predict(await exampleCanvas(ex))).probs);
  return state.liveProbs.get(ex.file);
}

// ---------- hero ----------

function renderHero() {
  $('#hero-bins').innerHTML = bins();
}

// ---------- 1 spel ----------

function newGame() {
  const normal = shuffle(state.data.examples.filter((e) => !e.note));
  const hard = shuffle(state.data.examples.filter((e) => e.note));
  const perClass = state.data.classes.map((_, k) => normal.find((e) => e.label === k)).filter(Boolean);
  const rounds = shuffle([...perClass, ...hard.slice(0, ROUNDS - perClass.length)]);
  state.game = { rounds, i: 0, phase: 'ask', score: { you: 0, cnn: 0, mobilenet: 0 }, log: [] };
  renderGame();
}

function renderScoreboard() {
  const g = state.game;
  const done = g.log.length;
  const item = (id, label, value) => `<div class="score ${id}"><b>${value}</b><span><i></i>${esc(label)}</span></div>`;
  $('#scoreboard').innerHTML = `
    ${item('you', 'Jij', g.score.you)}
    ${item('cnn', 'Eigen CNN', g.score.cnn)}
    ${item('mobilenet', 'MobileNet', g.score.mobilenet)}
    <div class="rounds" aria-label="Ronde ${Math.min(g.i + 1, g.rounds.length)} van ${g.rounds.length}">${g.rounds
      .map((_, i) => `<i class="${i < done ? 'done' : i === g.i ? 'now' : ''}"></i>`).join('')}</div>`;
}

function renderGame() {
  const g = state.game;
  renderScoreboard();
  const game = $('#game');
  if (g.i >= g.rounds.length) return renderGameEnd();
  const ex = g.rounds[g.i];
  const entry = g.log[g.i];

  let marks = {};
  let result = '';
  let actions = '';
  if (entry) {
    const chipsAt = {};
    const add = (k, html) => (chipsAt[k] ||= []).push(html);
    add(entry.you, `<span class="chip you">Jij</span>`);
    for (const m of MODELS) add(entry.picks[m.id], chip(m, entry.probs[m.id][entry.picks[m.id]]));
    marks = Object.fromEntries(state.data.classes.map((_, k) => [k, { chips: chipsAt[k], truth: k === ex.label, dim: !chipsAt[k] && k !== ex.label }]));
    const line = (who, k, p) => {
      const ok = k === ex.label;
      return `<span><b>${esc(who)}</b>: ${esc(cls(k))}${p !== undefined ? `, ${fmtPct(p)} zeker` : ''} <span class="${ok ? 'ok' : 'no'}">${ok ? '✓' : '✗'}</span></span>`;
    };
    result = [line('Jij', entry.you), ...MODELS.map((m) => line(m.name, entry.picks[m.id], entry.probs[m.id][entry.picks[m.id]]))].join('');
    const last = g.i === g.rounds.length - 1;
    actions = `<button type="button" class="button" id="next">${last ? 'Bekijk de uitslag' : 'Volgende foto →'}</button>
      <a class="button quiet" href="#machine">Hoe keek het CNN?</a>`;
  }

  game.innerHTML = `
    <figure class="card-photo">
      <img src="${exampleUrl(ex)}" alt="Foto ${g.i + 1} van ${g.rounds.length}">
      ${entry && ex.note ? `<span class="tag">${esc(ex.note)}</span>` : ''}
      <figcaption class="fig-cap">FIG. ${g.i + 1}/${g.rounds.length} — testfoto · 320 × 240 px${entry ? ` · ${esc(ex.file)}` : ''}</figcaption>
    </figure>
    <div class="game-side">
      <p class="game-q">${entry ? (entry.you === ex.label ? 'Goed gesorteerd.' : `Het was ${esc(cls(ex.label))}.`) : 'In welke bak hoort dit?'}</p>
      ${bins({ interactive: !entry, marks })}
      <div class="game-result">${result || '<span class="fine">Tik op een bak.</span>'}</div>
      <div class="game-actions">${actions}</div>
    </div>`;

  game.querySelectorAll('[data-pick]').forEach((b) => b.addEventListener('click', () => pick(Number(b.dataset.pick))));
  $('#next')?.addEventListener('click', () => {
    g.i += 1;
    renderGame();
    $('#spel').scrollIntoView({ block: 'start' });
  });
}

async function pick(k) {
  const g = state.game;
  if (g.log[g.i]) return;
  const ex = g.rounds[g.i];
  const probs = await exampleProbs(ex);
  const picks = Object.fromEntries(MODELS.map((m) => [m.id, argmax(probs[m.id])]));
  g.log[g.i] = { you: k, picks, probs };
  g.score.you += k === ex.label ? 1 : 0;
  for (const m of MODELS) g.score[m.id] += picks[m.id] === ex.label ? 1 : 0;
  renderGame();
  selectExample(ex); // de andere hoofdstukken kijken mee naar deze foto
}

function renderGameEnd() {
  const g = state.game;
  const n = g.rounds.length;
  const disagree = g.log.filter((e) => e.picks.cnn !== e.picks.mobilenet).length;
  const best = Math.max(g.score.cnn, g.score.mobilenet);
  const verdict = g.score.you > best ? 'Je wint' : g.score.you === best ? 'Gelijkspel' : 'De AI wint';
  const sureWrong = g.log.flatMap((e, i) => MODELS.filter((m) => e.picks[m.id] !== g.rounds[i].label && e.probs[m.id][e.picks[m.id]] >= SURE)).length;
  $('#game').innerHTML = `<div class="game-end">
    <p class="giant">${verdict}</p>
    <p class="lead">Jij ${g.score.you}, eigen CNN ${g.score.cnn}, MobileNet ${g.score.mobilenet} van de ${n}. De twee AI's waren het ${disagree} keer oneens${sureWrong ? `, en ${sureWrong} keer zat een AI er overtuigd naast (minstens 80% zeker)` : ''}.</p>
    <div class="game-actions">
      <button type="button" class="button" id="again">Nog een keer</button>
      <a class="button quiet" href="#machine">Kijk in het hoofd van de AI</a>
    </div>
  </div>`;
  $('#again').addEventListener('click', newGame);
}

// ---------- eigen foto ----------

function setupUpload() {
  const input = $('#upload');
  input.addEventListener('change', () => handleFile(input.files[0]));
  const zone = $('#dropzone');
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
  });
}

function showUploadError(text) {
  const err = $('#upload-error');
  err.textContent = text;
  err.hidden = !text;
}

async function handleFile(file) {
  if (!file) return;
  if (!state.sessions) return showUploadError('Eigen foto\'s werken alleen als de modellen geladen zijn.');
  if (!file.type.startsWith('image/')) return showUploadError('Dit is geen afbeelding. Kies een foto (bijvoorbeeld JPG, PNG of WebP).');
  if (file.size > MAX_UPLOAD_BYTES) return showUploadError('Deze foto is groter dan 20 MB. Kies een kleinere.');
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    showUploadError('');
    await setSource({ img, name: file.name, label: null, example: null });
    renderOwnResult();
  } catch {
    showUploadError('Deze foto kon niet worden geopend. Probeer een JPG of PNG.');
  } finally {
    URL.revokeObjectURL(url);
    $('#upload').value = '';
  }
}

function renderOwnResult() {
  const box = $('#own-result');
  const probs = state.base.probs;
  const chipsAt = {};
  for (const m of MODELS) (chipsAt[argmax(probs[m.id])] ||= []).push(chip(m, Math.max(...probs[m.id])));
  const marks = Object.fromEntries(state.data.classes.map((_, k) => [k, { chips: chipsAt[k], dim: !chipsAt[k] }]));
  box.hidden = false;
  box.innerHTML = `<canvas aria-label="Jouw foto"></canvas>
    <div class="game-side">${bins({ marks })}<p class="fine">Niemand weet hier het juiste antwoord. Klopt het? Dat beoordeel jij.</p></div>`;
  drawCanvas(box.querySelector('canvas'), state.source.base);
}

// ---------- gekozen foto: gedeeld door hoofdstuk 2 en 5 ----------

function buildStrips() {
  for (const holder of document.querySelectorAll('[data-thumbs]')) {
    for (const ex of state.data.examples) {
      const b = el('button', { type: 'button', 'aria-pressed': 'false', 'data-file': ex.file, 'data-tip': esc(`${cap(cls(ex.label))}${ex.note ? `: ${ex.note.toLowerCase()}` : ''}`), 'aria-label': cap(cls(ex.label)) },
        `<img src="${exampleUrl(ex)}" alt="" loading="lazy">`);
      b.addEventListener('click', () => selectExample(ex));
      holder.append(b);
    }
  }
}

async function selectExample(ex) {
  const img = await loadImage(exampleUrl(ex));
  await setSource({ img, name: ex.file, label: ex.label, example: ex });
}

async function setSource({ img, name, label, example }) {
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const base = canvasOf(w, h);
  base.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, w, h);
  state.source = { base, name, label, example };
  for (const t of document.querySelectorAll('[data-thumbs] button')) t.setAttribute('aria-pressed', String(!!example && t.dataset.file === name));

  state.base = await predict(base, example);
  drawCanvas($('#stage-photo'), base);
  renderPixels(state.base.crops.cnn);
  renderFeatureMaps(state.base.tensors.cnn);
  $('#stage-verdict').innerHTML = state.base.probs.cnn ? vbars(state.base.probs.cnn) : '<p class="fine">Modellen niet geladen.</p>';
  updateTest();
}

// ---------- 2 machine ----------

function rgbToImageData(rgb, size) {
  const img = new ImageData(size, size);
  for (let i = 0, j = 0; i < size * size; i++, j += 3) {
    img.data.set([rgb[j], rgb[j + 1], rgb[j + 2], 255], i * 4);
  }
  return img;
}

// Kanaalkleuren zijn functioneel: ze laten zien welk getal bij welke kleur hoort
const CHANNELS = [
  { name: 'rood', short: 'R', color: '#e03131' },
  { name: 'groen', short: 'G', color: '#2f9e44' },
  { name: 'blauw', short: 'B', color: '#1c7ed6' },
];

function renderPixels(crop) {
  const size = state.data.image_size.cnn;
  $('#pixel-canvas').getContext('2d').putImageData(rgbToImageData(crop, size), 0, 0);
  state.pixelCrop = crop;
  const [px, py] = state.pixel || [size / 2, size / 2];
  showPixel(px, py);
}

function pixelHover(e) {
  const size = state.data.image_size.cnn;
  const rect = $('#pixel-canvas').getBoundingClientRect();
  showPixel(
    Math.min(size - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * size))),
    Math.min(size - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * size))),
  );
}

function showPixel(px, py) {
  const crop = state.pixelCrop;
  if (!crop) return;
  state.pixel = [px, py];
  const size = state.data.image_size.cnn;
  const i = (py * size + px) * 3;
  const values = [crop[i], crop[i + 1], crop[i + 2]];

  const marker = $('#pixel-marker');
  marker.style.left = `${(px / size) * 100}%`;
  marker.style.top = `${(py / size) * 100}%`;
  marker.style.width = marker.style.height = `${100 / size}%`;

  $('#pixel-pos').innerHTML = `Pixel <b>x ${px}, y ${py}</b>`;
  $('#pixel-channels').innerHTML = `
    <div class="ch ch-head"><span>kanaal</span><span>0 – 255</span><span></span><span>model krijgt</span></div>
    ${CHANNELS.map((ch, c) => `
      <div class="ch">
        <span class="ch-name"><i style="background:${ch.color}"></i>${ch.name}</span>
        <span class="ch-bar"><i style="width:${(values[c] / 255) * 100}%;background:${ch.color}"></i></span>
        <b class="mono">${values[c]}</b>
        <span class="mono ch-model">${fmtNum((values[c] / 255 - MEAN[c]) / STD[c], 2)}</span>
      </div>`).join('')}`;

  const patch = new ImageData(7, 7);
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      const x = Math.min(size - 1, Math.max(0, px + dx));
      const y = Math.min(size - 1, Math.max(0, py + dy));
      const s = (y * size + x) * 3;
      patch.data.set([crop[s], crop[s + 1], crop[s + 2], 255], ((dy + 3) * 7 + dx + 3) * 4);
    }
  }
  $('#pixel-zoom').getContext('2d').putImageData(patch, 0, 0);
}

// Echte eerste laag van het CNN: 3x3-convolutie (BatchNorm ingevouwen) + ReLU
function renderFeatureMaps(tensor) {
  const { shape, weight, bias } = state.data.conv1;
  const [O, I, K] = shape;
  const size = state.data.image_size.cnn;
  const n = size * size;
  const grid = $('#feature-maps');
  if (!grid.children.length) {
    for (let o = 0; o < O; o++) {
      const b = el('button', { type: 'button', class: 'fmap', 'aria-pressed': 'false', 'aria-label': `Filter ${o + 1}` },
        `<canvas width="${size}" height="${size}"></canvas><span>#${String(o + 1).padStart(2, '0')}</span>`);
      b.addEventListener('click', () => selectFilter(o));
      grid.append(b);
    }
  }
  state.maps = [];
  const out = new Float32Array(n);
  for (let o = 0; o < O; o++) {
    let max = 0;
    let active = 0;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let s = bias[o];
        for (let i = 0; i < I; i++) {
          for (let ky = 0; ky < K; ky++) {
            const yy = y + ky - 1;
            if (yy < 0 || yy >= size) continue;
            for (let kx = 0; kx < K; kx++) {
              const xx = x + kx - 1;
              if (xx < 0 || xx >= size) continue;
              s += weight[((o * I + i) * K + ky) * K + kx] * tensor[i * n + yy * size + xx];
            }
          }
        }
        const v = s > 0 ? s : 0;
        out[y * size + x] = v;
        if (v > max) max = v;
        if (v > 0) active += 1;
      }
    }
    const img = new ImageData(size, size);
    for (let p = 0; p < n; p++) {
      const g = max > 0 ? Math.round((out[p] / max) * 255) : 0;
      img.data.set([g, g, g, 255], p * 4);
    }
    grid.children[o].querySelector('canvas').getContext('2d').putImageData(img, 0, 0);
    state.maps.push({ img, active: active / n });
  }
  // Eerste keer: toon het filter dat op deze foto het breedst reageert
  if (state.filter === undefined) state.filter = argmax(state.maps.map((m) => m.active));
  selectFilter(state.filter);
}

// Grote kaart + de echte 3x3-gewichten per kleurkanaal van één filter
function selectFilter(o) {
  state.filter = o;
  const { shape, weight } = state.data.conv1;
  const [, I, K] = shape;
  const size = state.data.image_size.cnn;
  const map = state.maps[o];
  for (const [k, b] of [...$('#feature-maps').children].entries()) b.setAttribute('aria-pressed', String(k === o));

  const w = (i, ky, kx) => weight[((o * I + i) * K + ky) * K + kx];
  const all = Array.from({ length: I * K * K }, (_, j) => Math.abs(weight[o * I * K * K + j]));
  const maxAbs = Math.max(...all) || 1;
  const kernels = CHANNELS.map((ch, i) => `
    <div class="kernel">
      <p class="kernel-name"><i style="background:${ch.color}"></i>${ch.name}</p>
      <div class="kernel-grid">${Array.from({ length: K * K }, (_, j) => {
        const v = w(i, Math.floor(j / K), j % K);
        const a = (Math.abs(v) / maxAbs) * 0.85;
        // + = donker vlak (hoe sterker, hoe donkerder), − = gestippelde rand
        const style = v >= 0
          ? `background:rgba(13,13,13,${a.toFixed(2)});color:${a > 0.45 ? '#ffffff' : 'var(--ink)'}`
          : `border:1px dashed rgba(13,13,13,${(0.25 + a * 0.75).toFixed(2)});color:var(--ink-2)`;
        return `<span style="${style}">${v >= 0 ? '+' : '−'}${fmtNum(Math.abs(v), 2)}</span>`;
      }).join('')}</div>
    </div>`).join('');

  const detail = $('#filter-detail');
  detail.innerHTML = `
    <p class="filter-title">Filter #${String(o + 1).padStart(2, '0')}</p>
    <canvas width="${size}" height="${size}" class="pixelated filter-big" aria-label="Kaart van filter ${o + 1}"></canvas>
    <p class="fine">Reageert op <b>${fmtPct(map.active)}</b> van de pixels van deze foto${map.active === 0 ? ': dit patroon zit er niet in' : ''}.</p>
    <p class="fine">Het patroon: 3 × 3 gewichten per kleur. <b>+</b> donker vlak = die kleur op die plek maakt het filter sterker, <b>−</b> gestippeld = zwakker.</p>
    <div class="kernels">${kernels}</div>`;
  detail.querySelector('canvas').getContext('2d').putImageData(map.img, 0, 0);
}

// ---------- 3 data ----------

function renderBinFill() {
  const { classes, trashnet_folders: f } = state.data;
  const counts = [f.paper + f.cardboard, f.glass, f.plastic, f.metal, f.trash];
  const COLS = 20;
  const S = 6;
  const GAP = 2;
  const maxRows = Math.ceil(Math.max(...counts) / COLS);
  const dpr = window.devicePixelRatio || 1;
  const holder = $('#bin-fill');
  holder.innerHTML = '';
  classes.forEach((c, k) => {
    const col = el('div', { class: 'fill-col', 'data-tip': esc(`<b>${cap(c)}</b><br>${fmtNum(counts[k])} foto's`) });
    const W = COLS * (S + GAP) - GAP;
    const H = maxRows * (S + GAP) - GAP;
    const canvas = canvasOf(W * dpr, H * dpr);
    canvas.style.aspectRatio = `${W} / ${H}`;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.fillStyle = cssVar(`--bin-${k}`);
    for (let i = 0; i < counts[k]; i++) {
      const r = Math.floor(i / COLS);
      const x = (i % COLS) * (S + GAP);
      const y = H - (r + 1) * (S + GAP) + GAP;
      ctx.fillRect(x, y, S, S);
    }
    col.innerHTML = `<p class="fill-num">${fmtNum(counts[k])}</p>`;
    col.append(canvas);
    col.insertAdjacentHTML('beforeend', `<div class="bin" data-k="${k}"><span class="bin-lid"></span><span class="bin-body"><span class="bin-label">${esc(cap(c))}</span></span></div>`);
    holder.append(col);
  });

  const sc = state.data.split_counts;
  const [tr, va, te] = ['train', 'val', 'test'].map((s) => sum(sc[s]));
  const all = tr + va + te;
  $('#split').innerHTML = `
    <div class="s-train" style="flex:${tr / all}" data-tip="Hiervan leert het model">${fmtNum(tr)} leren</div>
    <div class="s-val" style="flex:${va / all}" data-tip="Hiermee kozen we de beste trainingsronde">${fmtNum(va)} kiezen</div>
    <div class="s-test" style="flex:${te / all}" data-tip="De eindtoets: één keer gebruikt, aan het eind">${fmtNum(te)} toetsen</div>`;
}

function renderCurves() {
  const metric = state.curveMetric;
  const isAcc = metric === 'accuracy';
  modelLegend($('#curve-legend'));
  const container = $('#chart-curves');
  const W = Math.max(300, container.clientWidth);
  const H = 280;
  const m = { top: 14, right: 120, bottom: 30, left: 44 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const n = Math.max(...MODELS.map((mod) => state.history[mod.id].history.length));
  const values = MODELS.flatMap((mod) => state.history[mod.id].history.map((e) => e.val[metric]));
  const yMax = isAcc ? 1 : niceTicks(Math.max(...values)).at(-1);
  const yTicks = isAcc ? [0, 0.25, 0.5, 0.75, 1] : niceTicks(yMax);
  const x = (epoch) => m.left + ((epoch - 1) / (n - 1)) * iw;
  const y = (v) => m.top + ih - (v / yMax) * ih;
  const fmtY = (v) => (isAcc ? fmtPct(v) : fmtNum(v, 1));

  const parts = MODELS.map((mod) => {
    const h = state.history[mod.id];
    const d = h.history.map((e, i) => `${i ? 'L' : 'M'}${x(e.epoch).toFixed(1)},${y(e.val[metric]).toFixed(1)}`).join('');
    const best = h.history[h.best_epoch - 1];
    const last = h.history.at(-1);
    return `${linePath(mod.id, d)}
      ${marker(mod.id, x(best.epoch), y(best.val[metric]))}
      <text class="label-text" x="${x(last.epoch) + 8}" y="${y(last.val[metric]) + (mod.id === 'cnn' ? 14 : -6)}">${esc(mod.name)}</text>`;
  }).join('');

  const xTicks = [...new Set(niceTicks(n, 5).map((t) => Math.max(1, t)))];
  container.innerHTML = `<svg width="${W}" height="${H}" role="img" aria-label="Leercurves op de validatiefoto's">
    ${yTicks.map((t) => `<line class="grid-line" x1="${m.left}" x2="${m.left + iw}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-text" x="${m.left - 8}" y="${y(t) + 4}" text-anchor="end">${fmtY(t)}</text>`).join('')}
    ${xTicks.map((t) => `<text class="axis-text" x="${x(t)}" y="${H - 8}" text-anchor="middle">${t}</text>`).join('')}
    <text class="axis-text" x="${m.left + iw}" y="${H - 8}" dx="40">ronde</text>
    ${parts}
    <line class="crosshair" id="curve-cross" x1="0" x2="0" y1="${m.top}" y2="${m.top + ih}" visibility="hidden"/>
    <rect data-crosshair x="${m.left}" y="${m.top}" width="${iw}" height="${ih}" fill="transparent"/>
  </svg>`;

  const svg = container.querySelector('svg');
  const cross = svg.querySelector('#curve-cross');
  const overlay = svg.querySelector('[data-crosshair]');
  overlay.addEventListener('pointermove', (e) => {
    const rect = svg.getBoundingClientRect();
    const epoch = Math.min(n, Math.max(1, Math.round(((e.clientX - rect.left - m.left) / iw) * (n - 1) + 1)));
    cross.setAttribute('x1', x(epoch));
    cross.setAttribute('x2', x(epoch));
    cross.setAttribute('visibility', 'visible');
    const rows = MODELS.map((mod) => {
      const ep = state.history[mod.id].history[epoch - 1];
      return ep ? `${esc(mod.name)}: ${fmtY(ep.val[metric])} <span style="opacity:.7">(train ${fmtY(ep.train[metric])})</span>` : `${esc(mod.name)}: klaar na ${state.history[mod.id].epochs} rondes`;
    });
    showTip(`<b>Ronde ${epoch}</b><br>${rows.join('<br>')}`, e.clientX, e.clientY);
  });
  overlay.addEventListener('pointerleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
}

// ---------- 4 resultaten ----------

const binChip = (k) => `<span class="bin-chip"><i style="background:${binColor(k)}"></i>${esc(cls(k))}</span>`;

// Per model: één rij per soort, eerst goed (kleur), dan fout (open), dan overtuigd fout (rood)
function renderExam() {
  $('#exam').innerHTML = MODELS.map((mod) => {
    const e = state.evaluation[mod.id];
    const rows = state.data.classes.map((c, k) => {
      const correct = e.confusion_matrix[k][k];
      const total = sum(e.confusion_matrix[k]);
      const wrong = e.mistakes.filter((x) => x.true === c).sort((a, b) => a.confidence - b.confidence);
      const squares = Array.from({ length: correct }, () => `<i class="sq" style="--c:${binColor(k)}" data-tip="${esc(`<b>${cap(c)}</b>: goed`)}"></i>`).join('')
        + wrong.map((x) => `<i class="sq wrong${x.confidence >= SURE ? ' sure' : ''}" style="--c:${binColor(k)}" data-tip="${esc(`<b>${cap(x.true)}</b>, model zei <b>${x.predicted}</b><br>${fmtPct(x.confidence)} zeker · ${x.file.split('/')[1]}`)}"></i>`).join('');
      return `<div class="exam-row">
        <span class="exam-cls">${binChip(k)}</span>
        <div class="squares">${squares}</div>
        <span class="exam-score mono">${correct}/${total}</span>
      </div>`;
    }).join('');
    return `<div class="exam-model">
      <p class="exam-name ${mod.id}"><i></i>${esc(mod.name)}</p>
      <p class="exam-num">${fmtPct(e.accuracy, 1)}</p>
      <p class="exam-sub">${e.mistakes.length} fout van ${e.n_test} · ${e.confident_mistakes} daarvan minstens 80% zeker</p>
      <div class="exam-rows">${rows}</div>
    </div>`;
  }).join('');
}

// Recall per soort, beide modellen naast elkaar: dicht = eigen CNN, gearceerd = MobileNet
function renderClassBars() {
  $('#class-legend').innerHTML = `<span class="key"><i class="cb-key cnn"></i>Eigen CNN</span><span class="key"><i class="cb-key mobilenet"></i>MobileNet</span>`;
  $('#class-bars').innerHTML = state.data.classes.map((_, k) => `
    <div class="cb-row">
      <span class="cb-name">${binChip(k)}</span>
      <div class="cb-bars">${MODELS.map((m) => {
        const r = state.evaluation[m.id].per_class[k].recall;
        return `<div class="cb-bar ${m.id}" data-tip="${esc(`<b>${m.name}</b>: ${fmtPct(r)} van de ${cls(k)}-foto's goed`)}">
          <span class="cb-track"><i style="width:${(r * 100).toFixed(1)}%"></i></span><b class="mono">${fmtPct(r)}</b></div>`;
      }).join('')}</div>
    </div>`).join('');
}

// De vier grootste verwarringen per model, als leesbare zin met een balk (aandeel van die soort)
function renderMixups() {
  $('#mixups').innerHTML = MODELS.map((m) => {
    const e = state.evaluation[m.id];
    const pairs = [];
    e.confusion_matrix.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v > 0) pairs.push({ i, j, v, n: sum(row) }); }));
    pairs.sort((a, b) => b.v - a.v);
    return `<div class="mix-col">
      <p class="exam-name ${m.id}"><i></i>${esc(m.name)}</p>
      ${pairs.slice(0, 4).map(({ i, j, v, n }) => `
        <div class="mix-row">
          <b class="mix-count">${v}×</b>
          <p class="mix-text">${binChip(i)} <span class="mix-arrow">aangezien voor</span> ${binChip(j)}</p>
          <span class="mix-bar"><i style="width:${((v / n) * 100).toFixed(1)}%"></i></span>
          <span class="mix-share fine">${v} van de ${n} ${esc(cls(i))}-foto's (${fmtPct(v / n)})</span>
        </div>`).join('')}
    </div>`;
  }).join('');
}

function renderClassTable() {
  const rows = state.data.classes.map((c, i) => {
    const [a, b] = MODELS.map((m) => state.evaluation[m.id].per_class[i]);
    return [cap(c), fmtNum(a.support), fmtNum(a.precision, 2), fmtNum(a.recall, 2), fmtNum(a.f1, 2), fmtNum(b.precision, 2), fmtNum(b.recall, 2), fmtNum(b.f1, 2)];
  });
  $('#table-classes').innerHTML = table(['Soort', 'Testfoto\'s', 'CNN precisie', 'CNN recall', 'CNN F1', 'MobileNet precisie', 'MobileNet recall', 'MobileNet F1'], rows)
    + '<p class="fine" style="margin-top:8px">Precisie: als het model "glas" zegt, hoe vaak klopt dat? Recall: van alle glazen voorwerpen, hoeveel vond het?</p>';
}

// ---------- 5 lab ----------

function buildSliders() {
  const holder = $('#sliders');
  for (const s of SLIDERS) {
    const wrap = el('div', { class: 'slider' });
    wrap.innerHTML = `<label for="s-${s.id}"><span>${esc(s.label)}</span><span id="v-${s.id}">${esc(s.format(s.neutral))}</span></label>
      <input type="range" id="s-${s.id}" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.neutral}" disabled>`;
    holder.append(wrap);
    const input = wrap.querySelector('input');
    input.addEventListener('input', () => {
      state.params[s.id] = Number(input.value);
      $(`#v-${s.id}`).textContent = s.format(state.params[s.id]);
      updateTest();
    });
  }
  $('#reset').addEventListener('click', () => {
    for (const s of SLIDERS) {
      state.params[s.id] = s.neutral;
      $(`#s-${s.id}`).value = s.neutral;
      $(`#v-${s.id}`).textContent = s.format(s.neutral);
    }
    updateTest();
  });
}

function setLive(enabled) {
  for (const s of SLIDERS) $(`#s-${s.id}`).disabled = !enabled;
  $('#reset').disabled = !enabled;
  document.querySelector('label[for="upload"]').setAttribute('aria-disabled', String(!enabled));
  $('#upload').disabled = !enabled;
}

const isNeutral = () => SLIDERS.every((s) => state.params[s.id] === s.neutral);

// Zelfde volgorde als variations.py: licht, draaien, vervagen, kleiner op donker
function applyVariations(base, p) {
  const w = base.width;
  const h = base.height;
  let c = canvasOf(w, h);
  let ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(base, 0, 0);
  if (p.licht !== 1) {
    const d = ctx.getImageData(0, 0, w, h);
    brightness(d.data, p.licht);
    ctx.putImageData(d, 0, 0);
  }
  if (p.draaien !== 0) {
    const r = canvasOf(w, h);
    const rc = r.getContext('2d');
    rc.fillStyle = '#ffffff';
    rc.fillRect(0, 0, w, h);
    rc.translate(w / 2, h / 2);
    rc.rotate((-p.draaien * Math.PI) / 180); // Pillow draait tegen de klok in
    rc.drawImage(c, -w / 2, -h / 2);
    c = r;
    ctx = c.getContext('2d', { willReadFrequently: true });
  }
  if (p.vervagen > 0) {
    const d = ctx.getImageData(0, 0, w, h);
    boxBlur(d.data, w, h, (p.vervagen * w) / REFERENCE_WIDTH);
    ctx.putImageData(d, 0, 0);
  }
  if (p.tafel > 0) {
    const s = canvasOf(w, h);
    const sc = s.getContext('2d');
    sc.fillStyle = `rgb(${DARK.join(',')})`;
    sc.fillRect(0, 0, w, h);
    const sw = Math.round(w * (1 - p.tafel));
    const sh = Math.round(h * (1 - p.tafel));
    sc.drawImage(c, Math.floor((w - sw) / 2), Math.floor((h - sh) / 2), sw, sh);
    c = s;
  }
  return c;
}

function renderVerdicts(result, label) {
  const holder = $('#test-verdicts');
  if (!Object.keys(result.probs).length) {
    holder.innerHTML = '<div class="verdict"><p>Zonder geladen modellen kan deze foto niet beoordeeld worden.</p></div>';
    return;
  }
  holder.innerHTML = MODELS.map((m) => {
    const p = result.probs[m.id];
    const k = argmax(p);
    let badge = '';
    if (label !== null && label !== undefined) badge = k === label ? '<span class="badge good">✓ klopt</span>' : '<span class="badge bad">✗ fout</span>';
    return `<div class="verdict">
      <div class="verdict-head ${m.id}"><i></i>${esc(m.name)}${badge}</div>
      <div class="verdict-label" style="--c:${binColor(k)}"><span class="swatch"></span>${esc(cap(cls(k)))} <span class="verdict-pct">${fmtPct(p[k])}</span></div>
      ${vbars(p)}
    </div>`;
  }).join('');
}

// Nieuwe stand terwijl er nog gerekend wordt: daarna nog één keer met de laatste stand
async function updateTest() {
  if (!state.source) return;
  if (state.testBusy) {
    state.testPending = true;
    return;
  }
  state.testBusy = true;
  try {
    do {
      state.testPending = false;
      const varied = applyVariations(state.source.base, state.params);
      const result = isNeutral() ? state.base : await predict(varied, null);
      drawCanvas($('#test-image'), varied);
      renderVerdicts(result, state.source.label);
    } while (state.testPending);
  } finally {
    state.testBusy = false;
  }
}

function renderVariations() {
  modelLegend($('#var-legend'));
  const container = $('#chart-variations');
  container.innerHTML = '';
  const exps = state.variations.experiments;
  // Eerst alle panelen plaatsen, dan meten: met auto-fit krijgt een eenzaam paneel de volle breedte
  const holders = exps.map((exp) => {
    const block = el('div');
    block.append(el('h4', {}, esc(cap(exp.name))));
    const holder = el('div');
    block.append(holder);
    container.append(block);
    return holder;
  });
  for (const [k, exp] of exps.entries()) {
    const holder = holders[k];
    const W = Math.max(220, holder.clientWidth);
    const H = 170;
    const m = { top: 10, right: 18, bottom: 26, left: 40 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const n = exp.results.length;
    const x = (i) => m.left + (i / (n - 1)) * iw;
    const y = (v) => m.top + ih - v * ih;
    const neutralIdx = exp.results.findIndex((r) => r.level === exp.neutral);
    const fmtLevel = (v) => fmtNum(v, Number.isInteger(v) ? 0 : 2);
    holder.innerHTML = `<svg width="${W}" height="${H}" role="img" aria-label="${esc(exp.name)}">
      ${[0, 0.5, 1].map((t) => `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-text" x="${m.left - 6}" y="${y(t) + 4}" text-anchor="end">${fmtPct(t)}</text>`).join('')}
      <line class="crosshair" x1="${x(neutralIdx)}" x2="${x(neutralIdx)}" y1="${m.top}" y2="${m.top + ih}" stroke-dasharray="3 3"/>
      ${exp.results.map((r, i) => `<text class="axis-text" x="${x(i)}" y="${H - 6}" text-anchor="middle"${i === neutralIdx ? ' style="font-weight:700;fill:var(--ink)"' : ''}>${fmtLevel(r.level)}</text>`).join('')}
      ${MODELS.map((mod) => linePath(mod.id, exp.results.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(r.accuracy[mod.id])}`).join(''))).join('')}
      ${MODELS.map((mod) => exp.results.map((r, i) => marker(mod.id, x(i), y(r.accuracy[mod.id]), 4)).join('')).join('')}
      ${exp.results.map((r, i) => `<rect x="${x(i) - iw / (n - 1) / 2}" y="${m.top}" width="${iw / (n - 1)}" height="${ih}" fill="transparent" data-tip="${esc(`<b>${cap(exp.name)}: ${fmtLevel(r.level)} ${exp.unit}</b>${i === neutralIdx ? ' (origineel)' : ''}<br>${MODELS.map((mod) => `${mod.name} ${fmtPct(r.accuracy[mod.id], 1)}`).join('<br>')}`)}"/>`).join('')}
    </svg>`;
  }
  const get = (name, level) => exps.find((e) => e.name === name).results.find((r) => r.level === level).accuracy;
  const base = get('licht', 1);
  const dark = get('licht', 0.5);
  $('#var-note').textContent = `De verticale stippellijn is de originele foto. Voorbeeld: op half licht zakt het eigen CNN van ${fmtPct(base.cnn, 1)} naar ${fmtPct(dark.cnn, 1)} goed, MobileNet blijft op ${fmtPct(dark.mobilenet, 1)}. Ondersteboven gaat beter dan schuin: bij het trainen werden foto's gespiegeld, en twee keer spiegelen is 180° draaien.`;
}

// ---------- 6 zeker ≠ juist ----------

function renderMoment() {
  const c = state.evaluation.cnn;
  const mb = state.evaluation.mobilenet;
  const worst = c.mistakes[0];
  const ex = state.data.examples.find((e) => e.source === worst.file);
  if (ex) $('#moment-img').src = exampleUrl(ex);
  const share = (e) => e.confident_mistakes / e.mistakes.length;
  const k = (name) => state.data.classes.indexOf(name);
  $('#moment-stats').innerHTML = `
    <div><dt>zekerheid</dt><dd class="moment-pct">${fmtPct(worst.confidence)}</dd></div>
    <div><dt>eigen CNN zei</dt><dd>${binChip(k(worst.predicted))}</dd></div>
    <div><dt>het was</dt><dd>${binChip(k(worst.true))}</dd></div>`;
  $('#moment-text').textContent = `${ex?.note ? `${ex.note}. ` : ''}Van alle fouten van het eigen CNN was ${fmtPct(share(c))} minstens 80% zeker. Bij MobileNet, het betere model: ${fmtPct(share(mb))}. Een fout antwoord ziet er net zo stellig uit als een goed.`;

  const holder = $('#hard');
  holder.innerHTML = '';
  for (const ex2 of state.data.examples.filter((e) => e.note)) {
    const rows = MODELS.map((m) => {
      const p = ex2.probabilities[m.id];
      const k = argmax(p);
      return `<span class="row ${m.id}"><i></i>${esc(m.short)}: ${esc(cls(k))} ${fmtPct(p[k])} ${k === ex2.label ? '<span class="v">✓</span>' : '<span class="x">✗</span>'}</span>`;
    }).join('');
    const card = el('button', { type: 'button', class: 'hard-card', 'data-tip': 'Open in het lab' },
      `<img src="${exampleUrl(ex2)}" alt="" loading="lazy"><div><b>${esc(ex2.note)}</b><span>Echt: ${esc(cls(ex2.label))}</span>${rows}</div>`);
    card.addEventListener('click', async () => {
      await selectExample(ex2);
      $('#lab').scrollIntoView();
    });
    holder.append(card);
  }
}

// ---------- 7 responsible AI ----------

function renderRai() {
  const c = state.evaluation.cnn;
  const mob = state.evaluation.mobilenet;
  const f = state.data.trashnet_folders;
  const exps = state.variations.experiments;
  const get = (name, level) => exps.find((e) => e.name === name).results.find((r) => r.level === level).accuracy.cnn;
  const rows = [
    { label: 'Wit papier (zoals getraind)', v: get('licht', 1), base: true },
    { label: 'Donkere tafel, foto 75%', v: get('kleiner op donker', 0.75) },
    { label: 'Half zoveel licht', v: get('licht', 0.5) },
  ];
  $('#rai-bias-bars').innerHTML = `<p class="rai-bars-title">Eigen CNN, % goed op 380 testfoto's</p>${rows.map((r) => `
    <div class="rai-bar${r.base ? ' base' : ''}"><span>${esc(r.label)}</span><b>${fmtPct(r.v, 1)}</b>
      <i style="width:${(r.v * 100).toFixed(1)}%"></i></div>`).join('')}`;
  $('#rai-bias').textContent = `Elke trainingsfoto: wit papier, één schoon voorwerp. Daarbuiten zakt het eigen CNN fors. En restafval had maar ${fmtNum(f.trash)} foto's, tegen ${fmtNum(f.paper + f.cardboard)} van papier en karton.`;

  const share = (e) => e.confident_mistakes / e.mistakes.length;
  $('#rai-trust-big').textContent = fmtPct(share(mob));
  $('#rai-trust').textContent = `van de fouten van MobileNet was minstens 80% zeker; bij het eigen CNN ${fmtPct(share(c))}. Een fout antwoord ziet er even stellig uit als een goed.`;

  const min = (id) => fmtNum(state.history[id].seconds / 60, 1);
  const mb = (id) => fmtNum(state.onnx.find((o) => o.model === id).onnx_bytes / 1e6, 1);
  $('#rai-green-big').textContent = `${min('cnn')} min`;
  $('#rai-green').textContent = `training van het eigen CNN op één videokaart; het model is ${mb('cnn')} MB en voorspelt op jouw apparaat. MobileNet trainde hier ${min('mobilenet')} min (${mb('mobilenet')} MB), maar leunt op een eerdere training met 1,2 miljoen foto's die je hier niet ziet.`;
}

// ---------- 8 vragen ----------

function renderFaq() {
  const f = state.data.trashnet_folders;
  const fill = {
    'n-images': fmtNum(sum(Object.values(f))),
    'n-test': fmtNum(sum(state.data.split_counts.test)),
    'n-trash': fmtNum(f.trash),
    'n-paper': fmtNum(f.paper + f.cardboard),
  };
  for (const node of document.querySelectorAll('[data-fill]')) node.textContent = fill[node.dataset.fill];
}

// ---------- controle browser tegen Python ----------

// Zelfde formule als synthetic() in src/export_web.py
function syntheticCanvas({ w, h }) {
  const img = new ImageData(w, h);
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      img.data.set([Math.floor((xx * 255) / (w - 1)), Math.floor((yy * 255) / (h - 1)), (xx * 7 + yy * 13) % 256, 255], (yy * w + xx) * 4);
    }
  }
  const c = canvasOf(w, h);
  c.getContext('2d', { willReadFrequently: true }).putImageData(img, 0, 0);
  return c;
}

const maxAbsDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

async function runParityCheck() {
  const node = $('#parity');
  // 1. Exact dezelfde pixels: test model en voorbewerking
  const syn = await predict(syntheticCanvas(state.data.synthetic), null);
  const exactPer = Object.fromEntries(MODELS.map((m) => [m.id, maxAbsDiff(syn.probs[m.id], state.data.synthetic.probabilities[m.id])]));
  const exactDiff = Math.max(...Object.values(exactPer));
  // 2. Echte JPEG-voorbeelden: daar zit ook het openen van de JPEG tussen
  let maxDiff = 0;
  let same = 0;
  let total = 0;
  for (const ex of state.data.examples) {
    const probs = await exampleProbs(ex);
    for (const m of MODELS) {
      maxDiff = Math.max(maxDiff, maxAbsDiff(probs[m.id], ex.probabilities[m.id]));
      same += argmax(probs[m.id]) === argmax(ex.probabilities[m.id]) ? 1 : 0;
      total += 1;
    }
  }
  // Float32-afronding verschilt per rekenbibliotheek; 0,1 procentpunt is ruim daarboven
  const ok = same === total && exactDiff < 1e-3;
  node.dataset.state = ok ? 'ok' : 'warn';
  node.innerHTML = `${ok ? 'Ja, vrijwel.' : 'Niet helemaal.'} Op precies dezelfde pixels is het grootste verschil ${fmtNum(exactPer.cnn * 100, 4)} procentpunt (eigen CNN) en ${fmtNum(exactPer.mobilenet * 100, 4)} (MobileNet): afronding van kommagetallen. Het verkleinen en uitsnijden gaat tot op de pixel gelijk, en dat wordt automatisch getest. Op de ${state.data.examples.length} voorbeeldfoto's is het antwoord ${same} van de ${total} keer hetzelfde, en de zekerheid verschilt tot ${fmtNum(maxDiff * 100, 1)} procentpunt. Dat komt doordat browsers een JPEG net iets anders kunnen openen dan Python; in onze test scheelden pixels 1 of 2 op 255. Onzichtbaar voor jou, merkbaar voor het model.`;
  document.body.dataset.parity = JSON.stringify({ exactPer, same, total, maxDiff });
}

// ---------- IDE: actieve tab volgt het scrollen ----------

function setupIde() {
  const tabs = new Map([...document.querySelectorAll('[data-tab]')].map((a) => [a.dataset.tab, a]));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const a of tabs.values()) a.removeAttribute('aria-current');
      const tab = tabs.get(entry.target.id);
      tab.setAttribute('aria-current', 'page');
    }
  }, { rootMargin: '-45% 0px -54% 0px' });
  for (const id of tabs.keys()) observer.observe(document.getElementById(id));
}

// ---------- start ----------

async function loadModels() {
  setStatus('loading', 'modellen laden…');
  try {
    if (!window.ort) throw new Error('ONNX Runtime Web niet geladen');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.numThreads = 1;
    const sessions = await Promise.all(MODELS.map((m) => ort.InferenceSession.create(`../model/${m.id}.onnx`)));
    state.sessions = Object.fromEntries(MODELS.map((m, i) => [m.id, sessions[i]]));
  } catch (err) {
    console.error(err);
    setStatus('fallback', 'offline: vooraf berekend');
    $('#parity').textContent = 'Niet gecontroleerd: de modellen zijn niet geladen.';
    setLive(false);
    return;
  }
  setStatus('ready', 'modellen geladen');
  setLive(true);
  if (state.source?.example) await selectExample(state.source.example);
  runParityCheck();
}

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!state.data) return;
    renderCurves();
    renderVariations();
  }, 150);
});

async function init() {
  setupIde();
  try {
    const [data, hc, hm, ec, em, variations, onnx] = await Promise.all([
      loadJSON('data/web.json'),
      loadJSON('../results/history_cnn.json'),
      loadJSON('../results/history_mobilenet.json'),
      loadJSON('../results/evaluation_cnn.json'),
      loadJSON('../results/evaluation_mobilenet.json'),
      loadJSON('../results/variations.json'),
      loadJSON('../results/parity_onnx.json'),
    ]);
    Object.assign(state, { data, variations, onnx, history: { cnn: hc, mobilenet: hm }, evaluation: { cnn: ec, mobilenet: em } });
  } catch (err) {
    console.error(err);
    setStatus('error', 'open via een webserver');
    return;
  }

  renderHero();
  newGame();
  setupUpload();
  buildStrips();
  renderBinFill();
  segmented($('#curve-metric'), [{ id: 'accuracy', label: 'Goed' }, { id: 'loss', label: 'Fout-score (loss)' }], state.curveMetric, (id) => { state.curveMetric = id; renderCurves(); });
  renderCurves();
  renderExam();
  renderClassBars();
  renderMixups();
  renderClassTable();
  buildSliders();
  setLive(false);
  renderVariations();
  renderMoment();
  renderRai();
  renderFaq();
  $('#pixel-canvas').addEventListener('pointermove', pixelHover);
  $('#pixel-canvas').addEventListener('pointerdown', pixelHover); // ook tikken op een telefoon

  const first = state.data.examples.find((e) => !e.note && cls(e.label) === 'glas') || state.data.examples[0];
  await selectExample(first);
  loadModels();
}

init();
