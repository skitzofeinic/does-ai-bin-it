// Wat bakt AI ervan? Sorteerspel en explainer.
//
// Twee pagina's, één script: index.html is het spel (body data-page="spel"), explainer.html
// de uitleg (data-page="explainer"). Alle cijfers komen uit bestanden die de Python-scripts
// schreven (explainer/data/web.json en results/*.json); voorspellingen komen live uit model/*.onnx.
// Lukt het laden van ONNX Runtime niet, dan gebruikt de pagina de vooraf in Python
// berekende kansen voor de voorbeeldfoto's en schakelt uploads en schuifjes uit.

import { preprocess, toRGB, brightness, boxBlur, MEAN, STD, DARK, REFERENCE_WIDTH } from './preprocess.js?v=2026-11-02';
import { STRINGS, HTML_EN, ATTR_EN, NOTES_EN } from './i18n.js?v=2026-11-09b';

const GAME = document.body.dataset.page === 'spel';

const ORT_VERSION = '1.20.1';
const MAX_UPLOAD_BYTES = 20e6;
const MAX_SIDE = 1024; // grote uploads eerst verkleinen, anders wordt het schalen traag
const ROUNDS = 8;
const SURE = 0.8;

// DABI (Does AI Bin It) is het eigen CNN, in twee versies; MobileNet is voorgetraind, ter vergelijking.
// id = bestandsnaam (model/<id>.onnx, results/*_<id>.json), css = klassenaam voor de vorm
const MODELS = [
  { id: 'cnn', css: 'dabi11', name: 'DABI 1.1', short: 'DABI 1.1' },
  { id: 'cnn-1.2', css: 'dabi12', name: 'DABI 1.2', short: 'DABI 1.2' },
  { id: 'mobilenet', css: 'mobilenet', name: 'MobileNet', short: 'MobileNet' },
];
// "Hoe het kijkt", het lab-voorbeeld en de Responsible-AI-cijfers gaan over de nieuwste DABI
const DABI = 'cnn-1.2';
const modelName = (id) => MODELS.find((m) => m.id === id).name;

const SLIDERS = [
  { id: 'licht', min: 0.25, max: 2, step: 0.05, neutral: 1, format: (v) => `${fmtNum(v, 2)}×` },
  { id: 'draaien', min: 0, max: 180, step: 5, neutral: 0, format: (v) => `${v}°` },
  { id: 'vervagen', min: 0, max: 16, step: 1, neutral: 0, format: (v) => t('blurFmt', v) },
  // Hoeveel donkere tafel om de foto heen: de foto wordt (1 - waarde) zo groot
  { id: 'tafel', min: 0, max: 0.65, step: 0.05, neutral: 0, format: (v) => t('tableFmt', v > 0 ? Math.round((1 - v) * 100) : null) },
];
for (const sl of SLIDERS) Object.defineProperty(sl, 'label', { get: () => t(`sl_${sl.id}`) });

const state = {
  lang: 'nl',
  status: 'initial',
  uploadError: null,
  own: null,
  parity: null,
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
  return x.toLocaleString(state.lang === 'en' ? 'en-GB' : 'nl-NL', { minimumFractionDigits: digits, maximumFractionDigits: digits });
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
// Tekst in de gekozen taal; een functie in STRINGS krijgt de invulwaarden
function t(key, ...args) {
  const v = STRINGS[state.lang][key];
  return typeof v === 'function' ? v(...args) : v;
}
// Klassen heten in de data (web.json, evaluation_*.json) Nederlands; op het scherm in de gekozen taal
const cls = (k) => t('classes')[k];
const clsOf = (dutchName) => cls(state.data.classes.indexOf(dutchName));
const noteOf = (ex) => (state.lang === 'en' && NOTES_EN[ex.file]) || ex.note;
const binColor = (k) => `var(--bin-${k})`;

// De modellen kennen vijf materialen, Nederland heeft er vier bakken voor: plastic en metaal (blik)
// gaan samen in PMD. Per bak de materialen (indexen in state.data.classes); de kleur is die van het
// eerste materiaal. Het spel rekent per bak: hoe zeker een model van PMD is, is plastic + metaal.
const BINS = [[0], [1], [2, 3], [4]];
const binOf = (material) => BINS.findIndex((mats) => mats.includes(material));
const binName = (b) => t('bins')[b];
const binProbs = (probs) => BINS.map((mats) => sum(mats.map((k) => probs[k])));
// Score per bak zoals het spel rekent (kansen opgeteld per bak): results/evaluation_bins_<id>.json
const binAccuracy = (id) => state.binEval[id].accuracy;
// Hoe vaak een model plastic en metaal verwisselde: fout als soort, goed als bak (PMD)
const pmdSwaps = (e) => e.confusion_matrix[2][3] + e.confusion_matrix[3][2];
// Foute soort, maar wel de juiste bak (dus goed in het spel)?
const rightBinOnly = (k, label) => k !== label && binOf(k) === binOf(label);
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

function setStatus(stateName) {
  state.status = stateName;
  const s = $('#status');
  s.dataset.state = stateName === 'initial' ? 'loading' : stateName;
  s.textContent = t(`st_${stateName}`);
}

// Paden zijn relatief aan de pagina (index.html of explainer.html), niet aan dit script
const exampleUrl = (ex) => `assets/examples/${ex.file}`;
// De uitleg opent met deze foto in hoofdstuk 2 en 5
const explainerUrl = (ex, hash = '#machine') => `explainer.html${ex ? `?foto=${encodeURIComponent(ex.file)}` : ''}${hash}`;

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

// Goed = groen vierkant, fout = rode stip; de tekst erachter is voor schermlezers
const dot = (ok) => `<i class="dot ${ok ? 'ok' : 'no'}" aria-hidden="true"></i><span class="sr">${ok ? t('srOk') : t('srWrong')}</span>`;

const chip = (model, p) => `<span class="chip ${model.css}"><i></i>${esc(model.short)} ${fmtPct(p)}</span>`;

// Vier bakken (BINS). marks[b] = { chips: [html], truth, dim }; interactive maakt er knoppen van
function bins({ interactive = false, marks = {}, small = false } = {}) {
  return `<div class="bins" style="--n:${BINS.length}">${BINS.map((mats, b) => {
    const c = binName(b);
    const m = marks[b] || {};
    const tag = interactive ? 'button' : 'div';
    const attrs = interactive ? ` type="button" data-pick="${b}" aria-label="${esc(cap(c))}"` : '';
    return `<${tag} class="bin${m.truth ? ' truth' : ''}${m.dim ? ' dim' : ''}" data-k="${mats[0]}"${attrs}>
      <span class="chips">${(m.chips || []).join('')}</span>
      <span class="bin-lid"></span>
      <span class="bin-body"${small ? ' style="height:54px"' : ''}><span class="bin-label">${esc(cap(c))}</span></span>
    </${tag}>`;
  }).join('')}</div>`;
}

function vbars(probs) {
  const k = argmax(probs);
  return `<div class="vbars">${t('classes').map((c, i) => `
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
// DABI 1.2 = doorgetrokken lijn met dicht vierkant, DABI 1.1 = stippellijn met open vierkant,
// MobileNet = streepjeslijn met open rondje
const LINE = {
  'cnn': { color: 'var(--ink)', width: 1.5, dash: '2 3' },
  'cnn-1.2': { color: 'var(--ink)', width: 2, dash: '' },
  'mobilenet': { color: 'var(--ink)', width: 2, dash: '6 4' },
};
function marker(id, x, y, r = 5) {
  if (id === 'cnn-1.2') return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" fill="var(--ink)"/>`;
  if (id === 'cnn') return `<rect x="${x - r + 1}" y="${y - r + 1}" width="${2 * r - 2}" height="${2 * r - 2}" fill="var(--bg)" stroke="var(--ink)" stroke-width="2"/>`;
  return `<circle cx="${x}" cy="${y}" r="${r - 0.5}" fill="var(--bg)" stroke="var(--ink)" stroke-width="2"/>`;
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
  state.game = { rounds, i: 0, phase: 'ask', score: { you: 0, ...Object.fromEntries(MODELS.map((m) => [m.id, 0])) }, log: [] };
  $('#scoreboard').hidden = false;
  renderGame();
}

function renderScoreboard() {
  const g = state.game;
  const done = g.log.length;
  // Jij tegen de modellen; wie voorop staat (ook bij gelijkspel) krijgt een omgekeerd vak
  const best = Math.max(...Object.values(g.score));
  const fighter = (css, label, value) => `<div class="fighter ${css}${best > 0 && value === best ? ' lead' : ''}">
      <span class="f-name"><i></i>${esc(label)}</span><b>${value}</b></div>`;
  const round = Math.min(g.i + 1, g.rounds.length);
  $('#scoreboard').innerHTML = `
    <div class="versus">
      ${[fighter('you', t('you'), g.score.you), ...MODELS.map((m) => fighter(m.css, m.name, g.score[m.id]))].join('<span class="vs">vs</span>')}
    </div>
    <div class="rounds" aria-label="${t('roundAria', round, g.rounds.length)}">
      <span class="fine">${t('round', round, g.rounds.length)}</span>
      ${g.rounds.map((_, i) => `<i class="${i < done ? 'done' : i === g.i ? 'now' : ''}"></i>`).join('')}
    </div>`;
}

function renderGame() {
  const g = state.game;
  renderScoreboard();
  const game = $('#game');
  if (g.i >= g.rounds.length) return renderGameEnd();
  const ex = g.rounds[g.i];
  const entry = g.log[g.i];
  const truth = binOf(ex.label);

  let marks = {};
  let result = '';
  let actions = '';
  if (entry) {
    const chipsAt = {};
    const add = (b, html) => (chipsAt[b] ||= []).push(html);
    add(entry.you, `<span class="chip you">${esc(t('you'))}</span>`);
    for (const m of MODELS) add(entry.picks[m.id], chip(m, entry.probs[m.id][entry.picks[m.id]]));
    marks = Object.fromEntries(BINS.map((_, b) => [b, { chips: chipsAt[b], truth: b === truth, dim: !chipsAt[b] && b !== truth }]));
    const line = (who, b, p) => {
      const ok = b === truth;
      return `<span><b>${esc(who)}</b>: ${esc(binName(b))}${p !== undefined ? `, ${t('sure', fmtPct(p))}` : ''} ${dot(ok)}</span>`;
    };
    result = [line(t('you'), entry.you), ...MODELS.map((m) => line(m.name, entry.picks[m.id], entry.probs[m.id][entry.picks[m.id]]))].join('');
    const last = g.i === g.rounds.length - 1;
    actions = `<button type="button" class="button" id="next">${last ? t('seeResult') : t('nextPhoto')}</button>
      <a class="button quiet" href="${explainerUrl(ex)}">${t('howCnn')}</a>`;
  }

  game.innerHTML = `
    <figure class="card-photo">
      <img src="${exampleUrl(ex)}" alt="${t('photoAlt', g.i + 1, g.rounds.length)}">
      ${entry && ex.note ? `<span class="tag">${esc(noteOf(ex))}</span>` : ''}
      <figcaption class="fig-cap">${t('figCap', g.i + 1, g.rounds.length)}${entry ? ` · ${esc(ex.file)}` : ''}</figcaption>
    </figure>
    <div class="game-side">
      <p class="game-q${entry && entry.you === truth ? ' ok' : ''}">${entry
        ? (entry.you === truth ? t('goodSorted') : wasIn(ex.label))
        : t('gameQ')}</p>
      ${bins({ interactive: !entry, marks })}
      <div class="game-result">${result || `<span class="fine">${t('tapBin')}</span>`}</div>
      <div class="game-actions">${actions}</div>
    </div>`;

  game.querySelectorAll('[data-pick]').forEach((b) => b.addEventListener('click', () => pick(Number(b.dataset.pick))));
  $('#next')?.addEventListener('click', () => {
    g.i += 1;
    renderGame();
    $('#spel').scrollIntoView({ block: 'start' });
  });
}

// "Het was glas." of, als de bak anders heet dan het materiaal, "Het was metaal, dus PMD."
function wasIn(material) {
  const b = binOf(material);
  const hl = `<span class="hl" data-k="${BINS[b][0]}" style="--c:${binColor(BINS[b][0])}">${esc(binName(b))}</span>`;
  return BINS[b].length > 1 ? t('itWasIn', esc(cls(material)), hl) : t('itWas', hl);
}

// b = de gekozen bak; per model de bak met de hoogste kans, en die kans (som van de materialen)
async function pick(b) {
  const g = state.game;
  if (g.log[g.i]) return;
  const ex = g.rounds[g.i];
  const truth = binOf(ex.label);
  const probs = Object.fromEntries(Object.entries(await exampleProbs(ex)).map(([id, p]) => [id, binProbs(p)]));
  const picks = Object.fromEntries(MODELS.map((m) => [m.id, argmax(probs[m.id])]));
  g.log[g.i] = { you: b, picks, probs, truth };
  g.score.you += b === truth ? 1 : 0;
  for (const m of MODELS) g.score[m.id] += picks[m.id] === truth ? 1 : 0;
  renderGame();
}

const CROWN = '<svg class="crown" viewBox="0 0 24 16" aria-hidden="true"><path d="M2 15 1 3l6 5 5-7 5 7 6-5-1 12z"/></svg>';

// De AI's zijn het oneens als ze niet allemaal dezelfde bak kozen
const aisDisagree = (entry) => new Set(MODELS.map((m) => entry.picks[m.id])).size > 1;

// Podium: 2e links, 1e in het midden, dan 3e en 4e; bij gelijke score dezelfde plaats
function podium(players, n) {
  const ranked = players.map((p) => ({ ...p, rank: 1 + players.filter((o) => o.score > p.score).length }))
    .sort((a, b) => a.rank - b.rank);
  const order = [ranked[1], ranked[0], ...ranked.slice(2)];
  return `<ol class="podium" style="--n:${order.length}">${order.map((p) => `
    <li class="pod rank-${p.rank}${p.rank === 1 ? ' first' : ''}">
      ${p.rank === 1 ? `${CROWN}<span class="sr">${t('winner')}: </span>` : ''}
      <span class="p-name">${esc(p.name)}</span>
      <span class="p-score"><b class="mono">${p.score}</b> ${t('outOf', n)}</span>
      <span class="p-block">${t('place', p.rank)}</span>
    </li>`).join('')}</ol>`;
}

// Per foto (kolom) en per speler (rij): goed, fout of overtuigd fout; gearceerde kolom = de AI's oneens
function recap(players) {
  const g = state.game;
  const split = g.log.map(aisDisagree);
  const col = (i) => (split[i] ? ' rc-split' : '');
  const mark = (r) => `<i class="rc ${r.ok ? 'ok' : r.sure ? 'sure' : 'no'}" aria-hidden="true"></i><span class="sr">${r.ok ? t('srOk') : r.sure ? t('recapSure') : t('srWrong')}</span>`;
  return `<div class="recap" style="--n:${g.rounds.length}" role="table" aria-label="${t('recapAria')}">
    <div role="row"><span role="columnheader"></span>${g.rounds.map((ex, i) => `<span role="columnheader" class="rc-head${col(i)}">
      <img src="${exampleUrl(ex)}" alt="${esc(t('recapPhoto', i + 1, cls(ex.label)))}" data-tip="${esc(`${i + 1}. ${cap(cls(ex.label))}`)}"></span>`).join('')}</div>
    ${players.map((p) => `<div role="row"><span role="rowheader" class="rc-name">${esc(p.name)}</span>${p.results.map((r, i) => `<span role="cell" class="rc-cell${col(i)}">${mark(r)}</span>`).join('')}</div>`).join('')}
  </div>`;
}

function renderGameEnd() {
  const g = state.game;
  const n = g.rounds.length;
  const disagree = g.log.filter(aisDisagree).length;
  const best = Math.max(...MODELS.map((m) => g.score[m.id]));
  const [verdict, mood] = g.score.you > best ? [t('win'), 'win'] : g.score.you === best ? [t('draw'), 'draw'] : [t('aiWins'), 'lose'];
  const players = [
    { name: t('you'), score: g.score.you, results: g.log.map((e) => ({ ok: e.you === e.truth })) },
    ...MODELS.map((m) => ({
      name: m.name,
      score: g.score[m.id],
      results: g.log.map((e) => {
        const ok = e.picks[m.id] === e.truth;
        return { ok, sure: !ok && e.probs[m.id][e.picks[m.id]] >= SURE };
      }),
    })),
  ];
  const sureWrong = players.flatMap((p) => p.results).filter((r) => r.sure).length;
  // Het podium vervangt het scorebord, anders staat de stand er twee keer
  $('#scoreboard').hidden = true;
  $('#game').innerHTML = `<div class="game-end">
    <p class="giant end-verdict ${mood}">${verdict}</p>
    ${podium(players, n)}
    <h3 class="recap-title">${t('recapTitle')}</h3>
    ${recap(players)}
    <p class="rc-legend fine">
      <span><i class="rc ok"></i>${t('srOk')}</span>
      <span><i class="rc no"></i>${t('srWrong')}</span>
      <span><i class="rc sure"></i>${t('recapSureCount', sureWrong)}</span>
      <span><i class="rc-split-key"></i>${t('recapSplitCount', disagree)}</span>
    </p>
    <div class="game-actions">
      <button type="button" class="button" id="again">${t('again')}</button>
      <a class="button quiet" href="${explainerUrl(null)}">${t('lookInside')}</a>
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

function showUploadError(key) {
  state.uploadError = key;
  const err = $('#upload-error');
  err.textContent = key ? t(key) : '';
  err.hidden = !key;
}

async function handleFile(file) {
  if (!file) return;
  if (!state.sessions) return showUploadError('upNoModels');
  if (!file.type.startsWith('image/')) return showUploadError('upNotImage');
  if (file.size > MAX_UPLOAD_BYTES) return showUploadError('upTooBig');
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    showUploadError(null);
    await setSource({ img, name: file.name, label: null, example: null });
    state.own = { probs: state.base.probs, canvas: state.source.base };
    renderOwnResult();
  } catch {
    showUploadError('upBroken');
  } finally {
    URL.revokeObjectURL(url);
    $('#upload').value = '';
  }
}

function renderOwnResult() {
  if (!state.own) return;
  const box = $('#own-result');
  const chipsAt = {};
  for (const m of MODELS) {
    const p = binProbs(state.own.probs[m.id]);
    (chipsAt[argmax(p)] ||= []).push(chip(m, Math.max(...p)));
  }
  const marks = Object.fromEntries(BINS.map((_, b) => [b, { chips: chipsAt[b], dim: !chipsAt[b] }]));
  box.hidden = false;
  box.innerHTML = `<canvas aria-label="${t('ownAria')}"></canvas>
    <div class="game-side">${bins({ marks })}<p class="fine">${t('ownNote')}</p></div>`;
  drawCanvas(box.querySelector('canvas'), state.own.canvas);
}

// ---------- gekozen foto: gedeeld door hoofdstuk 2 en 5 ----------

function buildStrips() {
  for (const holder of document.querySelectorAll('[data-thumbs]')) {
    holder.innerHTML = '';
    for (const ex of state.data.examples) {
      const pressed = String(state.source?.example === ex);
      const b = el('button', { type: 'button', 'aria-pressed': pressed, 'data-file': ex.file, 'data-tip': `${cap(cls(ex.label))}${ex.note ? `: ${noteOf(ex).toLowerCase()}` : ''}`, 'aria-label': cap(cls(ex.label)) },
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
  if (GAME) return; // op de spelpagina alleen de kansen, voor de eigen foto
  drawCanvas($('#stage-photo'), base);
  renderPixels(state.base.crops[DABI]);
  renderFeatureMaps(state.base.tensors[DABI]);
  renderStageVerdict();
  updateTest();
}

function renderStageVerdict() {
  $('#stage-verdict').innerHTML = state.base.probs[DABI] ? vbars(state.base.probs[DABI]) : `<p class="fine">${t('modelsNotLoaded')}</p>`;
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
  { short: 'R', color: '#e03131' },
  { short: 'G', color: '#2f9e44' },
  { short: 'B', color: '#1c7ed6' },
];
CHANNELS.forEach((ch, c) => Object.defineProperty(ch, 'name', { get: () => t('channels')[c] }));

function renderPixels(crop) {
  const size = state.data.image_size[DABI];
  $('#pixel-canvas').getContext('2d').putImageData(rgbToImageData(crop, size), 0, 0);
  state.pixelCrop = crop;
  const [px, py] = state.pixel || [size / 2, size / 2];
  showPixel(px, py);
}

function pixelHover(e) {
  const size = state.data.image_size[DABI];
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
  const size = state.data.image_size[DABI];
  const i = (py * size + px) * 3;
  const values = [crop[i], crop[i + 1], crop[i + 2]];

  const marker = $('#pixel-marker');
  marker.style.left = `${(px / size) * 100}%`;
  marker.style.top = `${(py / size) * 100}%`;
  marker.style.width = marker.style.height = `${100 / size}%`;

  $('#pixel-pos').innerHTML = t('pixelPos', px, py);
  $('#pixel-channels').innerHTML = `
    <div class="ch ch-head"><span>${t('chHead')}</span><span>0 – 255</span><span></span><span>${t('modelGets')}</span></div>
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
  const { shape, weight, bias } = state.data.conv1[DABI];
  const [O, I, K] = shape;
  const size = state.data.image_size[DABI];
  const n = size * size;
  const grid = $('#feature-maps');
  if (!grid.children.length) {
    for (let o = 0; o < O; o++) {
      const b = el('button', { type: 'button', class: 'fmap', 'aria-pressed': 'false', 'aria-label': t('filterAria', o + 1) },
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
  const { shape, weight } = state.data.conv1[DABI];
  const [, I, K] = shape;
  const size = state.data.image_size[DABI];
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
        // + = gevuld vlak in de tekstkleur (hoe sterker, hoe voller), − = gestippelde rand; volgt het thema
        const mix = (share) => `color-mix(in srgb, var(--ink) ${(share * 100).toFixed(0)}%, transparent)`;
        const style = v >= 0
          ? `background:${mix(a)};color:${a > 0.45 ? 'var(--bg)' : 'var(--ink)'}`
          : `border:1px dashed ${mix(0.25 + a * 0.75)};color:var(--ink-2)`;
        return `<span style="${style}">${v >= 0 ? '+' : '−'}${fmtNum(Math.abs(v), 2)}</span>`;
      }).join('')}</div>
    </div>`).join('');

  const detail = $('#filter-detail');
  detail.innerHTML = `
    <p class="filter-title">Filter #${String(o + 1).padStart(2, '0')}</p>
    <canvas width="${size}" height="${size}" class="pixelated filter-big" aria-label="${t('filterMapAria', o + 1)}"></canvas>
    <p class="fine">${t('reacts', fmtPct(map.active), map.active === 0)}</p>
    <p class="fine">${t('pattern')}</p>
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
  classes.forEach((_, k) => {
    const c = cls(k);
    // el() zet attributen met setAttribute: niet escapen, anders staat <b> letterlijk in de tooltip
    const col = el('div', { class: 'fill-col', 'data-tip': t('fillTip', cap(c), fmtNum(counts[k])) });
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
    // Plastic en metaal: het materiaal op het etiket, en eronder in welke bak het in het spel gaat
    const pmd = BINS[binOf(k)].length > 1 ? `<span class="bin-sub">${esc(binName(binOf(k)))}</span>` : '';
    col.insertAdjacentHTML('beforeend', `<div class="bin" data-k="${k}"><span class="bin-lid"></span><span class="bin-body"><span class="bin-label">${esc(cap(c))}${pmd}</span></span></div>`);
    holder.append(col);
  });

  const sc = state.data.split_counts;
  const [tr, va, te] = ['train', 'val', 'test'].map((s) => sum(sc[s]));
  const all = tr + va + te;
  $('#split').innerHTML = `
    <div class="s-train" style="flex:${tr / all}" data-tip="${t('splitLearnTip')}">${t('splitLearn', fmtNum(tr))}</div>
    <div class="s-val" style="flex:${va / all}" data-tip="${t('splitChooseTip')}">${t('splitChoose', fmtNum(va))}</div>
    <div class="s-test" style="flex:${te / all}" data-tip="${t('splitTestTip')}">${t('splitTest', fmtNum(te))}</div>`;
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

  // Namen aan het eind van de lijn; liggen de eindpunten dicht bij elkaar, dan schuiven ze uit elkaar
  const labelY = {};
  const ends = MODELS.map((mod) => ({ id: mod.id, y: y(state.history[mod.id].history.at(-1).val[metric]) + 4 }))
    .sort((a, b) => a.y - b.y);
  ends.forEach((e, i) => { labelY[e.id] = i ? Math.max(e.y, labelY[ends[i - 1].id] + 13) : e.y; });

  const parts = MODELS.map((mod) => {
    const h = state.history[mod.id];
    const d = h.history.map((e, i) => `${i ? 'L' : 'M'}${x(e.epoch).toFixed(1)},${y(e.val[metric]).toFixed(1)}`).join('');
    const best = h.history[h.best_epoch - 1];
    const last = h.history.at(-1);
    return `${linePath(mod.id, d)}
      ${marker(mod.id, x(best.epoch), y(best.val[metric]))}
      <text class="label-text" x="${x(last.epoch) + 8}" y="${labelY[mod.id]}">${esc(mod.name)}</text>`;
  }).join('');

  // Wat de gekozen maat precies is; bij loss: welke loss, en hoe je het getal leest
  $('#curve-note').innerHTML = t(isAcc ? 'curveNoteAcc' : 'curveNoteLoss', fmtNum(sum(state.data.split_counts.val)));

  const xTicks = [...new Set(niceTicks(n, 5).map((t) => Math.max(1, t)))];
  container.innerHTML = `<svg width="${W}" height="${H}" role="img" aria-label="${t('curvesAria')}">
    ${yTicks.map((t) => `<line class="grid-line" x1="${m.left}" x2="${m.left + iw}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-text" x="${m.left - 8}" y="${y(t) + 4}" text-anchor="end">${fmtY(t)}</text>`).join('')}
    ${xTicks.map((t) => `<text class="axis-text" x="${x(t)}" y="${H - 8}" text-anchor="middle">${t}</text>`).join('')}
    <text class="axis-text" x="${m.left + iw}" y="${H - 8}" dx="40">${t('roundAxis')}</text>
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
      return ep ? `${esc(mod.name)}: ${fmtY(ep.val[metric])} <span style="opacity:.7">(train ${fmtY(ep.train[metric])})</span>` : `${esc(mod.name)}: ${t('doneAfter', state.history[mod.id].epochs)}`;
    });
    showTip(`<b>${t('roundTip', epoch)}</b><br>${rows.join('<br>')}`, e.clientX, e.clientY);
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
      const name = cap(cls(k));
      const correct = e.confusion_matrix[k][k];
      const total = sum(e.confusion_matrix[k]);
      const wrong = e.mistakes.filter((x) => x.true === c).sort((a, b) => a.confidence - b.confidence);
      const squares = Array.from({ length: correct }, () => `<i class="sq" style="--c:${binColor(k)}" data-tip="${esc(t('sqOk', name))}"></i>`).join('')
        + wrong.map((x) => `<i class="sq wrong${x.confidence >= SURE ? ' sure' : ''}" style="--c:${binColor(k)}" data-tip="${esc(t('sqWrong', name, clsOf(x.predicted), fmtPct(x.confidence), x.file.split('/')[1]))}"></i>`).join('');
      return `<div class="exam-row">
        <span class="exam-cls">${binChip(k)}</span>
        <div class="squares">${squares}</div>
        <span class="exam-score mono">${correct}/${total}</span>
      </div>`;
    }).join('');
    return `<div class="exam-model">
      <p class="exam-name ${mod.css}"><i></i>${esc(mod.name)}</p>
      <p class="exam-num">${fmtPct(e.accuracy, 1)}</p>
      <p class="exam-sub">${t('examSub', e.mistakes.length, e.n_test, e.confident_mistakes)}</p>
      <p class="exam-sub exam-bin">${t('examBin', fmtPct(binAccuracy(mod.id), 1), pmdSwaps(e))}</p>
      <div class="exam-rows">${rows}</div>
    </div>`;
  }).join('');
}

// Recall per soort, de drie modellen onder elkaar: grijs = DABI 1.1, zwart = DABI 1.2, gearceerd = MobileNet,
// en de naam voor elke balk, zodat je niet op de legenda hoeft te vertrouwen
function renderClassBars() {
  $('#class-legend').innerHTML = MODELS.map((m) => `<span class="key"><i class="cb-key ${m.css}"></i>${esc(m.name)}</span>`).join('');
  $('#class-bars').innerHTML = state.data.classes.map((_, k) => `
    <div class="cb-row">
      <span class="cb-name">${binChip(k)}</span>
      <div class="cb-bars">${MODELS.map((m) => {
        const r = state.evaluation[m.id].per_class[k].recall;
        return `<div class="cb-bar ${m.css}" data-tip="${esc(t('cbTip', m.name, fmtPct(r), cls(k)))}">
          <span class="cb-model">${esc(m.name)}</span><span class="cb-track"><i style="width:${(r * 100).toFixed(1)}%"></i></span><b class="mono">${fmtPct(r)}</b></div>`;
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
      <p class="exam-name ${m.css}"><i></i>${esc(m.name)}</p>
      ${pairs.slice(0, 4).map(({ i, j, v, n }) => `
        <div class="mix-row">
          <b class="mix-count">${v}×</b>
          <p class="mix-text">${binChip(i)} <span class="mix-arrow">${t('mixArrow')}</span> ${binChip(j)}</p>
          <span class="mix-bar"><i style="width:${((v / n) * 100).toFixed(1)}%"></i></span>
          <span class="mix-share fine">${t('mixShare', v, n, esc(cls(i)), fmtPct(v / n))}</span>
        </div>`).join('')}
    </div>`;
  }).join('');
}

// DABI 1.1 tegen 1.2 in één tabel: wat de nieuwe versie opleverde en wat het kostte
function renderVersions() {
  const [a, b] = ['cnn', 'cnn-1.2'].map((id) => state.evaluation[id]);
  const k = (name) => state.data.classes.indexOf(name);
  const plasticAsGlass = (e) => e.confusion_matrix[k('plastic')][k('glas')];
  const minutes = (id) => state.history[id] && state.history[id].seconds / 60;
  const mb = (id) => state.onnx.find((o) => o.model === id)?.onnx_bytes / 1e6;
  // higher: is een hoger getal beter? null = geen oordeel (grootte, rekentijd)
  const rows = [
    { label: t('verAccuracy'), a: a.accuracy, b: b.accuracy, fmt: (v) => fmtPct(v, 1), diff: (d) => t('pp', fmtNum(d * 100, 1)), higher: true },
    { label: t('verBalanced'), a: a.balanced_accuracy, b: b.balanced_accuracy, fmt: (v) => fmtPct(v, 1), diff: (d) => t('pp', fmtNum(d * 100, 1)), higher: true },
    { label: t('verBin'), a: binAccuracy('cnn'), b: binAccuracy('cnn-1.2'), fmt: (v) => fmtPct(v, 1), diff: (d) => t('pp', fmtNum(d * 100, 1)), higher: true, pct: true },
    { label: t('verPlasticGlass'), a: plasticAsGlass(a), b: plasticAsGlass(b), fmt: (v) => fmtNum(v), diff: (d) => fmtNum(d), higher: false },
    { label: t('verSure'), a: a.confident_mistakes, b: b.confident_mistakes, fmt: (v) => fmtNum(v), diff: (d) => fmtNum(d), higher: false },
    { label: t('verSize'), a: mb('cnn'), b: mb('cnn-1.2'), fmt: (v) => `${fmtNum(v, 1)} MB`, diff: (d) => `${fmtNum(d, 1)} MB`, higher: null },
    { label: t('verTime'), a: minutes('cnn'), b: minutes('cnn-1.2'), fmt: (v) => `${fmtNum(v, 1)} min`, diff: (d) => `${fmtNum(d, 1)} min`, higher: null },
  ].filter((r) => r.a !== undefined && r.b !== undefined && !Number.isNaN(r.a) && !Number.isNaN(r.b));
  // Halter per maat: open vierkant = 1.1, dicht = 1.2, de lijn ertussen groen (beter), rood (slechter)
  // of grijs (kosten). Elke maat begint bij 0; percentages lopen tot 100%, de rest tot iets boven het grootste getal.
  $('#version-legend').innerHTML = `<span class="key"><i class="vs-key v11"></i>DABI 1.1</span><span class="key"><i class="vs-key v12"></i>DABI 1.2</span>
    <span class="key"><i class="vs-key-line better"></i>${t('verBetter')}</span><span class="key"><i class="vs-key-line worse"></i>${t('verWorse')}</span>
    <span class="key"><i class="vs-key-line cost"></i>${t('verCost')}</span>`;
  $('#versions').innerHTML = rows.map((r) => {
    const d = r.b - r.a;
    const better = r.higher === null || Math.abs(d) < 1e-9 ? null : (d > 0) === r.higher;
    const mood = better === null ? 'cost' : better ? 'better' : 'worse';
    const max = r.pct || r.label === t('verAccuracy') || r.label === t('verBalanced') ? 1 : Math.max(r.a, r.b) * 1.25;
    const pos = (v) => ((v / max) * 100).toFixed(2);
    return `<div class="vs-row ${mood}">
      <span class="vs-label">${esc(r.label)}</span>
      <div class="vs-track" role="img" aria-label="${esc(`${r.label}: DABI 1.1 ${r.fmt(r.a)}, DABI 1.2 ${r.fmt(r.b)}`)}">
        <i class="vs-line" style="left:${pos(Math.min(r.a, r.b))}%;width:${(Math.abs(d) / max * 100).toFixed(2)}%"></i>
        <i class="vs-dot v11" style="left:${pos(r.a)}%" data-tip="${esc(`DABI 1.1: <b>${r.fmt(r.a)}</b>`)}"></i>
        <i class="vs-dot v12" style="left:${pos(r.b)}%" data-tip="${esc(`DABI 1.2: <b>${r.fmt(r.b)}</b>`)}"></i>
      </div>
      <span class="vs-vals mono">${r.fmt(r.a)} → <b>${r.fmt(r.b)}</b></span>
      <span class="vs-diff mono">${d > 0 ? '+' : d < 0 ? '−' : '±'}${r.diff(Math.abs(d))}</span>
    </div>`;
  }).join('');
}

function renderClassTable() {
  const rows = state.data.classes.map((_, i) => {
    const per = MODELS.map((m) => state.evaluation[m.id].per_class[i]);
    return [cap(cls(i)), fmtNum(per[0].support), ...per.flatMap((p) => [fmtNum(p.precision, 2), fmtNum(p.recall, 2), fmtNum(p.f1, 2)])];
  });
  const head = [...t('tableHead'), ...MODELS.flatMap((m) => t('tableCols').map((c) => `${m.name} ${c}`))];
  $('#table-classes').innerHTML = table(head, rows)
    + `<p class="fine" style="margin-top:8px">${esc(t('tableNote'))}</p>`;
}

// ---------- 5 lab ----------

function buildSliders() {
  const holder = $('#sliders');
  for (const s of SLIDERS) {
    const wrap = el('div', { class: 'slider' });
    wrap.innerHTML = `<label for="s-${s.id}"><span id="l-${s.id}">${esc(s.label)}</span><span id="v-${s.id}">${esc(s.format(s.neutral))}</span></label>
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
  if (GAME) {
    document.querySelector('label[for="upload"]').setAttribute('aria-disabled', String(!enabled));
    $('#upload').disabled = !enabled;
    return;
  }
  for (const s of SLIDERS) $(`#s-${s.id}`).disabled = !enabled;
  $('#reset').disabled = !enabled;
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
    holder.innerHTML = `<div class="verdict"><p>${t('verdictNoModels')}</p></div>`;
    return;
  }
  holder.innerHTML = MODELS.map((m) => {
    const p = result.probs[m.id];
    const k = argmax(p);
    let badge = '';
    if (label !== null && label !== undefined) {
      badge = k === label ? `<span class="badge good">${dot(true)}${t('badgeOk')}</span>`
        : rightBinOnly(k, label) ? `<span class="badge binonly">${dot(false)}${t('badgeBinOnly')}</span>`
          : `<span class="badge bad">${dot(false)}${t('badgeWrong')}</span>`;
    }
    return `<div class="verdict">
      <div class="verdict-head ${m.css}"><i></i>${esc(m.name)}${badge}</div>
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
    block.append(el('h4', {}, esc(cap(t('exp')[exp.name]))));
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
    holder.innerHTML = `<svg width="${W}" height="${H}" role="img" aria-label="${esc(t('exp')[exp.name])}">
      ${[0, 0.5, 1].map((t) => `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-text" x="${m.left - 6}" y="${y(t) + 4}" text-anchor="end">${fmtPct(t)}</text>`).join('')}
      <line class="crosshair" x1="${x(neutralIdx)}" x2="${x(neutralIdx)}" y1="${m.top}" y2="${m.top + ih}" stroke-dasharray="3 3"/>
      ${exp.results.map((r, i) => `<text class="axis-text" x="${x(i)}" y="${H - 6}" text-anchor="middle"${i === neutralIdx ? ' style="font-weight:700;fill:var(--ink)"' : ''}>${fmtLevel(r.level)}</text>`).join('')}
      ${MODELS.map((mod) => linePath(mod.id, exp.results.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(r.accuracy[mod.id])}`).join(''))).join('')}
      ${MODELS.map((mod) => exp.results.map((r, i) => marker(mod.id, x(i), y(r.accuracy[mod.id]), 4)).join('')).join('')}
      ${exp.results.map((r, i) => `<rect x="${x(i) - iw / (n - 1) / 2}" y="${m.top}" width="${iw / (n - 1)}" height="${ih}" fill="transparent" data-tip="${esc(`<b>${cap(t('exp')[exp.name])}: ${fmtLevel(r.level)} ${t('unit')[exp.unit]}</b>${i === neutralIdx ? t('original') : ''}<br>${MODELS.map((mod) => `${mod.name} ${fmtPct(r.accuracy[mod.id], 1)}`).join('<br>')}`)}"/>`).join('')}
    </svg>`;
  }
  const get = (name, level) => exps.find((e) => e.name === name).results.find((r) => r.level === level).accuracy;
  const base = get('licht', 1);
  const dark = get('licht', 0.5);
  $('#var-note').textContent = t('varNote', modelName(DABI), fmtPct(base[DABI], 1), fmtPct(dark[DABI], 1), fmtPct(dark.mobilenet, 1));
}

// "A, B en C"
const listOf = (items) => (items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')} ${t('and')} ${items.at(-1)}`);
const sureShare = (id) => state.evaluation[id].confident_mistakes / state.evaluation[id].mistakes.length;

// ---------- 6 zeker ≠ juist ----------

function renderMoment() {
  // De stelligste fout van de nieuwste DABI, op een foto die de site heeft (assets/examples)
  const mistakes = state.evaluation[DABI].mistakes;
  const worst = mistakes.find((x) => state.data.examples.some((e) => e.source === x.file)) || mistakes[0];
  const ex = state.data.examples.find((e) => e.source === worst.file);
  if (ex) $('#moment-img').src = exampleUrl(ex);
  $('#moment-img').alt = ex ? noteOf(ex) : t('momentAlt');
  const k = (name) => state.data.classes.indexOf(name);
  $('#moment-stats').innerHTML = `
    <div><dt>${t('dtSure')}</dt><dd class="moment-pct">${fmtPct(worst.confidence)}</dd></div>
    <div><dt>${t('dtSaid', modelName(DABI))}</dt><dd>${binChip(k(worst.predicted))}</dd></div>
    <div><dt>${t('dtWas')}</dt><dd>${binChip(k(worst.true))}</dd></div>`;
  $('#moment-text').textContent = t('momentText', ex?.note ? noteOf(ex) : '', listOf(MODELS.map((m) => `${m.name} ${fmtPct(sureShare(m.id))}`)));

  const holder = $('#hard');
  holder.innerHTML = '';
  for (const ex2 of state.data.examples.filter((e) => e.note)) {
    const rows = MODELS.map((m) => {
      const p = ex2.probabilities[m.id];
      const k = argmax(p);
      // Groene stip = goed, rode stip = fout; de tekst erachter is voor schermlezers
      const ok = k === ex2.label;
      const game = rightBinOnly(k, ex2.label) ? ` <span class="row-game">(${esc(t('pmdInGame'))})</span>` : '';
      return `<span class="row ${ok ? 'ok' : 'no'}"><i aria-hidden="true"></i><span class="sr">${ok ? t('srOk') : t('srWrong')}: </span>${esc(m.short)}: ${esc(cls(k))} ${fmtPct(p[k])}${game}</span>`;
    }).join('');
    const card = el('button', { type: 'button', class: 'hard-card', 'data-tip': t('openLab') },
      `<img src="${exampleUrl(ex2)}" alt="" loading="lazy"><div><b>${esc(noteOf(ex2))}</b><span>${esc(t('actually', cls(ex2.label)))}</span>${rows}</div>`);
    card.addEventListener('click', async () => {
      await selectExample(ex2);
      $('#lab').scrollIntoView();
    });
    holder.append(card);
  }
}

// ---------- 7 responsible AI ----------

function renderRai() {
  const f = state.data.trashnet_folders;
  const exps = state.variations.experiments;
  const get = (name, level) => exps.find((e) => e.name === name).results.find((r) => r.level === level).accuracy[DABI];
  const rows = [
    { label: t('raiRowWhite'), v: get('licht', 1), base: true },
    { label: t('raiRowDark'), v: get('kleiner op donker', 0.75) },
    { label: t('raiRowHalf'), v: get('licht', 0.5) },
  ];
  $('#rai-bias-bars').innerHTML = `<p class="rai-bars-title">${t('raiBarsTitle', modelName(DABI))}</p>${rows.map((r) => `
    <div class="rai-bar${r.base ? ' base' : ''}"><span>${esc(r.label)}</span><b>${fmtPct(r.v, 1)}</b>
      <i style="width:${(r.v * 100).toFixed(1)}%"></i></div>`).join('')}`;
  $('#rai-bias').textContent = t('raiBias', modelName(DABI), fmtNum(f.trash), fmtNum(f.paper + f.cardboard));
  $('#rai-bias-did').textContent = t('raiBiasDid');

  $('#rai-trust-big').textContent = fmtPct(sureShare('mobilenet'));
  $('#rai-trust').textContent = t('raiTrust', listOf(['cnn', 'cnn-1.2'].map((id) => t('forModel', modelName(id), fmtPct(sureShare(id))))));

  const min = (id) => fmtNum(state.history[id].seconds / 60, 1);
  const mb = (id) => fmtNum(state.onnx.find((o) => o.model === id).onnx_bytes / 1e6, 1);
  $('#rai-green-big').textContent = `${min(DABI)} min`;
  $('#rai-green').textContent = t('raiGreen', modelName(DABI), min('mobilenet'), mb('mobilenet'));
  $('#rai-green-did').textContent = t('raiGreenDid', mb(DABI));
}

// ---------- 8 vragen ----------

function renderFaq() {
  const f = state.data.trashnet_folders;
  const k = (name) => state.data.classes.indexOf(name);
  const plasticAsGlass = (id) => state.evaluation[id].confusion_matrix[k('plastic')][k('glas')];
  const varied = (name, level) => fmtPct(state.variations.experiments.find((e) => e.name === name).results.find((r) => r.level === level).accuracy[DABI]);
  const fill = {
    'n-images': fmtNum(sum(Object.values(f))),
    'n-test': fmtNum(sum(state.data.split_counts.test)),
    'n-trash': fmtNum(f.trash),
    'n-paper': fmtNum(f.paper + f.cardboard),
    'pg-11': fmtNum(plasticAsGlass('cnn')),
    'pg-12': fmtNum(plasticAsGlass('cnn-1.2')),
    'pg-mob': fmtNum(plasticAsGlass('mobilenet')),
    'n-plastic-test': fmtNum(sum(state.evaluation.cnn.confusion_matrix[k('plastic')])),
    'n-train': fmtNum(sum(state.data.split_counts.train)),
    'n-val': fmtNum(sum(state.data.split_counts.val)),
    'v-white': varied('licht', 1),
    'v-half': varied('licht', 0.5),
    'v-blur': varied('vervagen', 4),
    'v-dark': varied('kleiner op donker', 0.75),
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
  state.parity = { ok, exactPer, same, total, maxDiff };
  renderParity();
  document.body.dataset.parity = JSON.stringify({ exactPer, same, total, maxDiff });
}

function renderParity() {
  const node = $('#parity');
  const p = state.parity;
  if (!p) {
    node.textContent = state.status === 'fallback' ? t('parityNotChecked') : t('parityPending');
    return;
  }
  node.dataset.state = p.ok ? 'ok' : 'warn';
  node.textContent = t('parity', p.ok, listOf(MODELS.map((m) => `${fmtNum(p.exactPer[m.id] * 100, 4)} (${m.name})`)),
    state.data.examples.length, p.same, p.total, fmtNum(p.maxDiff * 100, 1));
}

// ---------- IDE: actieve tab volgt het scrollen ----------

// Schuift alleen de tabbalk zelf, niet de pagina (scrollIntoView zou ook verticaal scrollen)
function centerTab(tab) {
  const nav = tab.parentElement;
  const n = nav.getBoundingClientRect();
  const a = tab.getBoundingClientRect();
  const left = nav.scrollLeft + a.left - n.left - (n.width - a.width) / 2;
  nav.scrollTo({ left, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
}

function setupIde() {
  // Alleen tabs naar een hoofdstuk op deze pagina; de rest linkt naar de andere pagina
  const tabs = new Map([...document.querySelectorAll('[data-tab]')].map((a) => [a.dataset.tab, a]));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const a of tabs.values()) a.removeAttribute('aria-current');
      const tab = tabs.get(entry.target.id);
      tab.setAttribute('aria-current', 'page');
      centerTab(tab);
    }
  }, { rootMargin: '-45% 0px -54% 0px' });
  for (const id of tabs.keys()) observer.observe(document.getElementById(id));
}

// ---------- taal en thema ----------

function renderCurveMetric() {
  segmented($('#curve-metric'), [{ id: 'accuracy', label: t('metricRight') }, { id: 'loss', label: t('metricLoss') }], state.curveMetric, (id) => { state.curveMetric = id; renderCurves(); });
}

// De Nederlandse tekst staat in index.html; bewaar die voordat er iets vertaald wordt
const NL_HTML = new Map([...document.querySelectorAll('[data-i18n]')].map((n) => [n, n.innerHTML]));
const NL_ATTR = [...document.querySelectorAll('[data-i18n-attr]')].map((n) => {
  const [attr, key] = n.dataset.i18nAttr.split(':');
  return { n, attr, key, nl: n.getAttribute(attr) };
});

function applyLang() {
  const en = state.lang === 'en';
  for (const [n, nl] of NL_HTML) n.innerHTML = en ? HTML_EN[n.dataset.i18n] : nl;
  for (const { n, attr, key, nl } of NL_ATTR) n.setAttribute(attr, en ? ATTR_EN[key] : nl);
  document.documentElement.lang = state.lang;
  document.title = t(GAME ? 'docTitle' : 'docTitleExplainer');
  document.querySelector('meta[name="description"]').content = t(GAME ? 'docDesc' : 'docDescExplainer');
  const btn = $('#lang-toggle');
  btn.setAttribute('aria-label', t('langLabel'));
  for (const opt of btn.querySelectorAll('[data-lang]')) opt.classList.toggle('on', opt.dataset.lang === state.lang);
  updateThemeButton();
  setStatus(state.status);
}

function setLang(lang) {
  state.lang = lang;
  try { localStorage.setItem('lang', lang); } catch { /* alleen voor deze sessie */ }
  const url = new URL(location.href);
  url.searchParams.delete('lang');
  history.replaceState(null, '', url);
  hideTip();
  applyLang();
  if (!state.data) return;
  // Alles opnieuw tekenen; spelstand, gekozen foto en schuifjes blijven staan
  if (GAME) {
    renderGame();
    renderOwnResult();
    if (state.uploadError) showUploadError(state.uploadError);
    return;
  }
  renderHero();
  buildStrips();
  renderBinFill();
  renderCurveMetric();
  renderCurves();
  renderExam();
  renderClassBars();
  renderMixups();
  renderClassTable();
  renderVersions();
  for (const s of SLIDERS) {
    $(`#l-${s.id}`).textContent = s.label;
    $(`#v-${s.id}`).textContent = s.format(state.params[s.id]);
  }
  renderVariations();
  renderMoment();
  renderRai();
  renderFaq();
  renderParity();
  if (state.base) {
    showPixel(...state.pixel);
    for (const [o, b] of [...$('#feature-maps').children].entries()) b.setAttribute('aria-label', t('filterAria', o + 1));
    selectFilter(state.filter);
    renderStageVerdict();
    updateTest();
  }
}

function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  const btn = $('#theme-toggle');
  btn.setAttribute('aria-pressed', String(dark));
  btn.setAttribute('aria-label', t('theme'));
  btn.title = t('theme');
}

function setupToggles() {
  // ?lang=en opent direct in het Engels; anders de vorige keuze, anders Nederlands
  let saved = null;
  try { saved = localStorage.getItem('lang'); } catch { /* geen opslag */ }
  const asked = new URLSearchParams(location.search).get('lang');
  state.lang = STRINGS[asked] ? asked : STRINGS[saved] ? saved : 'nl';
  $('#lang-toggle').addEventListener('click', () => setLang(state.lang === 'nl' ? 'en' : 'nl'));

  // Het thema staat al op <html> (inline script in index.html, vóór het tekenen)
  $('#theme-toggle').addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('theme', theme); } catch { /* alleen voor deze sessie */ }
    document.querySelector('meta[name="theme-color"]').content = cssVar('--bg');
    updateThemeButton();
  });
  updateThemeButton();
}

// ---------- start ----------

async function loadModels() {
  setStatus('loading');
  try {
    if (!window.ort) throw new Error('ONNX Runtime Web niet geladen');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.numThreads = 1;
    const sessions = await Promise.all(MODELS.map((m) => ort.InferenceSession.create(`model/${m.id}.onnx`)));
    state.sessions = Object.fromEntries(MODELS.map((m, i) => [m.id, sessions[i]]));
  } catch (err) {
    console.error(err);
    setStatus('fallback');
    if (!GAME) renderParity();
    setLive(false);
    return;
  }
  setStatus('ready');
  setLive(true);
  if (GAME) return;
  if (state.source?.example) await selectExample(state.source.example);
  runParityCheck();
}

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!state.data || GAME) return;
    renderCurves();
    renderVariations();
  }, 150);
});

async function init() {
  setupIde();
  setupToggles();
  if (state.lang !== 'nl') applyLang();
  // Het spel heeft alleen de voorbeeldfoto's nodig
  if (GAME) {
    try {
      state.data = await loadJSON('explainer/data/web.json');
    } catch (err) {
      console.error(err);
      setStatus('error');
      return;
    }
    newGame();
    setupUpload();
    setLive(false);
    loadModels();
    return;
  }

  try {
    const [data, variations, onnx, histories, evaluations, binEvals] = await Promise.all([
      loadJSON('explainer/data/web.json'),
      loadJSON('results/variations.json'),
      loadJSON('results/parity_onnx.json'),
      Promise.all(MODELS.map((m) => loadJSON(`results/history_${m.id}.json`))),
      Promise.all(MODELS.map((m) => loadJSON(`results/evaluation_${m.id}.json`))),
      Promise.all(MODELS.map((m) => loadJSON(`results/evaluation_bins_${m.id}.json`))),
    ]);
    Object.assign(state, {
      data, variations, onnx,
      history: Object.fromEntries(MODELS.map((m, i) => [m.id, histories[i]])),
      evaluation: Object.fromEntries(MODELS.map((m, i) => [m.id, evaluations[i]])),
      binEval: Object.fromEntries(MODELS.map((m, i) => [m.id, binEvals[i]])),
    });
  } catch (err) {
    console.error(err);
    setStatus('error');
    return;
  }

  renderHero();
  buildStrips();
  renderBinFill();
  renderCurveMetric();
  renderCurves();
  renderExam();
  renderClassBars();
  renderMixups();
  renderClassTable();
  renderVersions();
  buildSliders();
  setLive(false);
  renderVariations();
  renderMoment();
  renderRai();
  renderFaq();
  $('#pixel-canvas').addEventListener('pointermove', pixelHover);
  $('#pixel-canvas').addEventListener('pointerdown', pixelHover); // ook tikken op een telefoon

  // ?foto=glass21.jpg komt van het spel: open met de foto die je net sorteerde
  const asked = new URLSearchParams(location.search).get('foto');
  const first = state.data.examples.find((e) => e.file === asked)
    || state.data.examples.find((e) => !e.note && state.data.classes[e.label] === 'glas') || state.data.examples[0];
  await selectExample(first);
  loadModels();
}

init();
