// Waste Recycle Classifier: interactieve explainer.
//
// Alle cijfers komen uit bestanden die de Python-scripts schreven (data/web.json en
// ../results/*.json); voorspellingen komen live uit de ONNX-modellen in ../model/.
// Lukt het laden van ONNX Runtime niet, dan toont de pagina de vooraf in Python
// berekende kansen voor de voorbeeldfoto's en schakelt uploads en schuifjes uit.

import { preprocess, toRGB, brightness, boxBlur, DARK, REFERENCE_WIDTH } from './preprocess.js';

const ORT_VERSION = '1.20.1';
const MAX_UPLOAD_BYTES = 20e6;
const MAX_SIDE = 1024; // grote uploads eerst verkleinen, anders wordt het schalen traag

const MODELS = [
  { id: 'cnn', name: 'Eigen CNN', detail: 'zelf getraind, vanaf nul', color: 'var(--series-1)' },
  { id: 'mobilenet', name: 'MobileNet', detail: 'voorgetraind, ter vergelijking', color: 'var(--series-2)' },
];

const SLIDERS = [
  { id: 'licht', label: 'Licht', min: 0.25, max: 2, step: 0.05, neutral: 1, format: (v) => `${fmtNum(v, 2)}× helderheid` },
  { id: 'draaien', label: 'Draaien', min: 0, max: 180, step: 5, neutral: 0, format: (v) => `${v}°` },
  { id: 'vervagen', label: 'Vervagen', min: 0, max: 16, step: 1, neutral: 0, format: (v) => (v ? `straal ${v}` : 'scherp') },
  { id: 'kleiner', label: 'Kleiner op donkere ondergrond', min: 0.35, max: 1, step: 0.05, neutral: 1, format: (v) => `${Math.round(v * 100)}% van het beeld` },
];

const state = {
  data: null,
  history: {},
  evaluation: {},
  variations: null,
  onnx: null,
  sessions: null,
  source: null,
  base: null,
  params: Object.fromEntries(SLIDERS.map((s) => [s.id, s.neutral])),
  curveModel: 'cnn',
  curveMetric: 'accuracy',
  cmModel: 'cnn',
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
// Nooit 100% tonen als het model niet precies 1 gaf
function fmtPct(x, digits = 0) {
  const factor = 10 ** digits;
  // toPrecision haalt float-ruis weg: 0.795 * 100 is 79.49999999999999
  let v = Math.round(Number((x * 100).toPrecision(12)) * factor) / factor;
  if (v >= 100 && x < 1) v = 100 - 1 / factor;
  return `${fmtNum(v, digits)}%`;
}
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const argmax = (arr) => arr.reduce((best, v, i) => (v > arr[best] ? i : best), 0);
const modelById = (id) => MODELS.find((m) => m.id === id);

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

function setStatus(stateName, text) {
  const s = $('#status');
  s.dataset.state = stateName;
  s.textContent = text;
}

// ---------- tooltip ----------

const tooltip = () => $('#tooltip');
function showTip(html, x, y) {
  const t = tooltip();
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
const hideTip = () => { tooltip().hidden = true; };

// Elementen met data-tip krijgen automatisch een tooltip
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

function legend(container, items) {
  container.innerHTML = items
    .map((it) => `<span><span class="${it.shape === 'dot' ? 'dot' : 'key'}" style="background:${it.color}"></span>${esc(it.label)}</span>`)
    .join('');
}

// Horizontale balken (HTML), één reeks. rows: {label, value, text, tip, top}
function hbars(rows, color, max = Math.max(...rows.map((r) => r.value))) {
  return rows
    .map((r) => `
      <div class="hbar${r.top ? ' top' : ''}"${r.tip ? ` data-tip="${esc(r.tip)}"` : ''}>
        <span>${esc(r.label)}</span>
        <span class="hbar-track">
          <span class="hbar-fill" style="width:${(r.value / max) * 85}%; --bar:${color}"></span>
          <span class="hbar-value">${esc(r.text)}</span>
        </span>
      </div>`)
    .join('');
}

function table(headers, rows) {
  return `<div class="table-scroll"><table class="data"><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows
    .map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join('')}</tr>`)
    .join('')}</tbody></table></div>`;
}

function niceTicks(max, count = 4) {
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = 0; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

// ---------- 2 Train: dataset ----------

function renderDataset() {
  const { classes, split_counts: sc } = state.data;
  const rows = classes.map((c, i) => {
    const total = sc.train[i] + sc.val[i] + sc.test[i];
    return {
      label: cap(c),
      value: total,
      text: fmtNum(total),
      tip: `<strong>${esc(cap(c))}</strong><br>train ${fmtNum(sc.train[i])} · validatie ${fmtNum(sc.val[i])} · test ${fmtNum(sc.test[i])}`,
    };
  });
  $('#chart-dataset').innerHTML = hbars(rows, 'var(--neutral-mark)');
  $('#table-dataset').innerHTML = table(
    ['Soort', 'Train', 'Validatie', 'Test', 'Totaal'],
    classes.map((c, i) => [cap(c), sc.train[i], sc.val[i], sc.test[i], sc.train[i] + sc.val[i] + sc.test[i]].map((v) => (typeof v === 'number' ? fmtNum(v) : v))),
  );
}

// ---------- 2 Train: leercurves ----------

function renderCurves() {
  const h = state.history[state.curveModel];
  const model = modelById(state.curveModel);
  const metric = state.curveMetric;
  const isAcc = metric === 'accuracy';
  const series = [
    { key: 'train', label: 'Trainingsfoto\'s', color: 'var(--neutral-mark)' },
    { key: 'val', label: 'Validatiefoto\'s', color: model.color },
  ];
  legend($('#curve-legend'), [...series.map((s) => ({ label: s.label, color: s.color })), { label: 'gekozen ronde', color: model.color, shape: 'dot' }]);

  const container = $('#chart-curves');
  const W = Math.max(280, container.clientWidth);
  const H = 260;
  const m = { top: 12, right: 16, bottom: 32, left: 44 };
  const iw = W - m.left - m.right;
  const ih = H - m.top - m.bottom;
  const n = h.history.length;
  const values = h.history.flatMap((e) => [e.train[metric], e.val[metric]]);
  const yMax = isAcc ? 1 : niceTicks(Math.max(...values)).at(-1);
  const yTicks = isAcc ? [0, 0.25, 0.5, 0.75, 1] : niceTicks(yMax);
  const x = (epoch) => m.left + ((epoch - 1) / Math.max(1, n - 1)) * iw;
  const y = (v) => m.top + ih - (v / yMax) * ih;
  const xTicks = niceTicks(n, 5).filter((t) => t >= 1 || t === 0).map((t) => Math.max(1, t));
  const fmtY = (v) => (isAcc ? fmtPct(v) : fmtNum(v, 1));

  const path = (key) => h.history.map((e, i) => `${i ? 'L' : 'M'}${x(e.epoch).toFixed(1)},${y(e[key][metric]).toFixed(1)}`).join('');
  const best = h.history[h.best_epoch - 1];

  container.innerHTML = `
    <svg width="${W}" height="${H}" role="img" aria-label="Leercurve ${esc(model.name)}">
      ${yTicks.map((t) => `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-text" x="${m.left - 8}" y="${y(t) + 4}" text-anchor="end">${fmtY(t)}</text>`).join('')}
      <line class="base-line" x1="${m.left}" x2="${W - m.right}" y1="${y(0)}" y2="${y(0)}"/>
      ${[...new Set(xTicks)].map((t) => `<text class="axis-text" x="${x(t)}" y="${H - 10}" text-anchor="middle">${t}</text>`).join('')}
      ${series.map((s) => `<path class="series-line" d="${path(s.key)}" style="stroke:${s.color}"/>`).join('')}
      <circle cx="${x(best.epoch)}" cy="${y(best.val[metric])}" r="5" fill="${model.color}" stroke="var(--surface)" stroke-width="2"/>
      <line class="crosshair" id="curve-cross" x1="0" x2="0" y1="${m.top}" y2="${m.top + ih}" visibility="hidden"/>
      <rect data-crosshair x="${m.left}" y="${m.top}" width="${iw}" height="${ih}" fill="transparent"/>
    </svg>`;

  const svg = container.querySelector('svg');
  const cross = svg.querySelector('#curve-cross');
  const overlay = svg.querySelector('[data-crosshair]');
  overlay.addEventListener('pointermove', (e) => {
    const rect = svg.getBoundingClientRect();
    const epoch = Math.min(n, Math.max(1, Math.round(((e.clientX - rect.left - m.left) / iw) * (n - 1) + 1)));
    const ep = h.history[epoch - 1];
    cross.setAttribute('x1', x(epoch));
    cross.setAttribute('x2', x(epoch));
    cross.setAttribute('visibility', 'visible');
    showTip(`<strong>Ronde ${epoch}</strong>${epoch === h.best_epoch ? ' (gekozen)' : ''}<br>train ${fmtY(ep.train[metric])}<br>validatie ${fmtY(ep.val[metric])}`, e.clientX, e.clientY);
  });
  overlay.addEventListener('pointerleave', () => {
    cross.setAttribute('visibility', 'hidden');
    hideTip();
  });

  const minutes = h.seconds / 60;
  $('#curve-caption').textContent = isAcc
    ? `Horizontaal: trainingsronde. ${model.name}: ${h.epochs} rondes in ${fmtNum(minutes, 1)} minuten op een RTX 2070. Gekozen: ronde ${h.best_epoch}, ${fmtPct(best.val.accuracy)} goed op de validatiefoto's. Trainingsfoto's worden bij elke ronde willekeurig aangepast, daarom ligt die lijn soms onder de validatielijn.`
    : `Fout-score (loss): hoe ver de zekerheid van het model naast het juiste antwoord zit. Lager is beter. Een fout met veel zekerheid kost meer dan een twijfelende fout.`;

  $('#table-curves').innerHTML = table(
    ['Ronde', 'Train', 'Validatie'],
    h.history.map((e) => [String(e.epoch), fmtY(e.train[metric]), fmtY(e.val[metric])]),
  );
}

// ---------- 2 Train: eindtoets ----------

function renderStatTiles() {
  $('#stat-tiles').innerHTML = MODELS.map((m) => {
    const e = state.evaluation[m.id];
    return `<div class="card model-card" style="--model:${m.color}">
      <p class="stat-label"><span class="swatch"></span>${esc(m.name)}: goed op de testset</p>
      <p class="stat-value">${fmtPct(e.accuracy, 1)}</p>
      <p class="stat-sub">${fmtNum(e.mistakes.length)} fouten op ${fmtNum(e.n_test)} foto's · gebalanceerd ${fmtPct(e.balanced_accuracy, 1)}</p>
    </div>`;
  }).join('');
}

// Lichte cel: bijna-zwarte tekst, donkere cel: witte tekst, in beide thema's (de ramp is gelijk)
const SEQ_STEPS = 8;
function renderConfusion() {
  const e = state.evaluation[state.cmModel];
  const classes = e.classes;
  const rows = e.confusion_matrix;
  const head = `<tr><th></th><th></th><th colspan="${classes.length}" class="axis-title">Model zei</th></tr>
    <tr><th></th><th></th>${classes.map((c) => `<th>${esc(cap(c))}</th>`).join('')}</tr>`;
  const body = rows.map((row, i) => {
    const total = row.reduce((a, b) => a + b, 0);
    return `<tr>${i === 0 ? `<th rowspan="${classes.length}" class="axis-title" style="writing-mode:vertical-rl;transform:rotate(180deg)">Echt</th>` : ''}<th>${esc(cap(classes[i]))}</th>${row
      .map((v, j) => {
        const share = total ? v / total : 0;
        const step = v === 0 ? 0 : Math.max(1, Math.min(SEQ_STEPS - 1, Math.ceil(share * (SEQ_STEPS - 1))));
        const ink = step >= 4 ? '#ffffff' : '#0b0b0b';
        const tip = `<strong>Echt ${esc(classes[i])}, model zei ${esc(classes[j])}</strong><br>${fmtNum(v)} foto's (${fmtPct(share)} van ${esc(classes[i])})${i === j ? '<br>goed' : ''}`;
        return `<td style="background:var(--seq-${step});color:${ink}" data-tip="${esc(tip)}">${v || ''}</td>`;
      })
      .join('')}</tr>`;
  }).join('');
  $('#chart-confusion').innerHTML = `<table aria-label="Confusion matrix ${esc(modelById(state.cmModel).name)}"><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

function renderClassTable() {
  const { classes } = state.data;
  const rows = classes.map((c, i) => {
    const [a, b] = MODELS.map((m) => state.evaluation[m.id].per_class[i]);
    return [cap(c), fmtNum(a.support), fmtNum(a.precision, 2), fmtNum(a.recall, 2), fmtNum(a.f1, 2), fmtNum(b.precision, 2), fmtNum(b.recall, 2), fmtNum(b.f1, 2)];
  });
  $('#table-classes').innerHTML = table(
    ['Soort', 'Testfoto\'s', 'CNN precisie', 'CNN recall', 'CNN F1', 'MobileNet precisie', 'MobileNet recall', 'MobileNet F1'],
    rows,
  );
}

// ---------- 4 Test: gemeten variaties ----------

function renderVariations() {
  legend($('#var-legend'), MODELS.map((m) => ({ label: m.name, color: m.color })));
  const container = $('#chart-variations');
  container.innerHTML = '';
  const exps = state.variations.experiments;
  // Eerst alle panelen plaatsen, dan meten: met auto-fit krijgt een eenzaam paneel de volle breedte
  const holders = exps.map((exp) => {
    const block = el('div');
    block.append(el('h4', {}, `${esc(cap(exp.name))} <span class="muted">(${esc(exp.unit)})</span>`));
    const holder = el('div');
    block.append(holder);
    container.append(block);
    return holder;
  });
  for (const [k, exp] of exps.entries()) {
    const holder = holders[k];
    const W = Math.max(220, holder.clientWidth);
    const H = 180;
    const m = { top: 10, right: 18, bottom: 28, left: 40 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const n = exp.results.length;
    const x = (i) => m.left + (i / (n - 1)) * iw;
    const y = (v) => m.top + ih - v * ih;
    const neutralIdx = exp.results.findIndex((r) => r.level === exp.neutral);
    const fmtLevel = (v) => fmtNum(v, Number.isInteger(v) ? 0 : 2);
    holder.innerHTML = `<svg width="${W}" height="${H}" role="img" aria-label="${esc(exp.name)}">
      ${[0, 0.5, 1].map((t) => `<line class="grid-line" x1="${m.left}" x2="${W - m.right}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-text" x="${m.left - 6}" y="${y(t) + 4}" text-anchor="end">${fmtPct(t)}</text>`).join('')}
      <line class="crosshair" x1="${x(neutralIdx)}" x2="${x(neutralIdx)}" y1="${m.top}" y2="${m.top + ih}"/>
      <text class="label-text" x="${x(neutralIdx) + (neutralIdx === n - 1 ? -4 : 4)}" y="${m.top + 10}" text-anchor="${neutralIdx === n - 1 ? 'end' : 'start'}">origineel</text>
      ${exp.results.map((r, i) => `<text class="axis-text" x="${x(i)}" y="${H - 8}" text-anchor="middle">${fmtLevel(r.level)}</text>`).join('')}
      ${MODELS.map((mod) => `<path class="series-line" style="stroke:${mod.color}" d="${exp.results.map((r, i) => `${i ? 'L' : 'M'}${x(i)},${y(r.accuracy[mod.id])}`).join('')}"/>`).join('')}
      ${MODELS.map((mod) => exp.results.map((r, i) => `<circle cx="${x(i)}" cy="${y(r.accuracy[mod.id])}" r="4" fill="${mod.color}" stroke="var(--surface)" stroke-width="2"/>`).join('')).join('')}
      ${exp.results.map((r, i) => `<rect x="${x(i) - iw / (n - 1) / 2}" y="${m.top}" width="${iw / (n - 1)}" height="${ih}" fill="transparent" data-tip="${esc(`<strong>${cap(exp.name)}: ${fmtLevel(r.level)} ${exp.unit}</strong><br>${MODELS.map((mod) => `${mod.name} ${fmtPct(r.accuracy[mod.id], 1)}`).join('<br>')}`)}"/>`).join('')}
    </svg>`;
  }

  const get = (name, level) => exps.find((e) => e.name === name).results.find((r) => r.level === level).accuracy;
  const base = get('licht', 1);
  const dark = get('licht', 0.5);
  const blur = get('vervagen', 4);
  const small = get('kleiner op donker', 0.75);
  $('#var-takeaways').innerHTML = [
    `Op half licht zakt het eigen CNN van ${fmtPct(base.cnn, 1)} naar ${fmtPct(dark.cnn, 1)} goed; MobileNet merkt er nauwelijks iets van (${fmtPct(base.mobilenet, 1)} → ${fmtPct(dark.mobilenet, 1)}).`,
    `Een beetje vervagen (straal 4) kost het CNN ${fmtNum((base.cnn - blur.cnn) * 100)} procentpunt en MobileNet ${fmtNum((base.mobilenet - blur.mobilenet) * 100)}.`,
    `Op een donkere ondergrond, met het voorwerp iets kleiner (75%), zakt het CNN al naar ${fmtPct(small.cnn, 1)} en MobileNet blijft op ${fmtPct(small.mobilenet, 1)}: het CNN leerde de witte achtergrond mee.`,
    rotationTakeaway(exps.find((e) => e.name === 'draaien'), base),
  ].map((t) => `<li>${esc(t)}</li>`).join('');

  $('#table-variations').innerHTML = table(
    ['Aanpassing', 'Stand', ...MODELS.map((m) => m.name)],
    exps.flatMap((e) => e.results.map((r) => [cap(e.name), `${fmtNum(r.level, Number.isInteger(r.level) ? 0 : 2)} ${e.unit}`, ...MODELS.map((m) => fmtPct(r.accuracy[m.id], 1))])),
  );
}

function rotationTakeaway(exp, base) {
  const drop = (id) => Math.max(...exp.results.map((r) => base[id] - r.accuracy[id]));
  const at = (deg) => exp.results.find((r) => r.level === deg).accuracy;
  let text = `Draaien kost het minst: hooguit ${fmtNum(drop('cnn') * 100)} procentpunt (CNN) en ${fmtNum(drop('mobilenet') * 100)} (MobileNet).`;
  if (MODELS.every((m) => at(180)[m.id] > at(45)[m.id])) {
    text += ' Ondersteboven (180°) gaat zelfs beter dan schuin (45°): tijdens het trainen werden foto\'s gespiegeld, en twee keer spiegelen is 180° draaien.';
  }
  return text;
}

// ---------- 5 Reflecteer: teksten met echte cijfers ----------

function renderReflection() {
  const [c, mb] = MODELS.map((m) => state.evaluation[m.id]);
  const share = (e) => e.confident_mistakes / e.mistakes.length;
  const worst = (e) => e.mistakes[0];
  $('#reflect-confidence').innerHTML = `Bij een fout is het model vaak nog steeds heel zeker. Het eigen CNN was ${fmtPct(worst(c).confidence)} zeker dat een foto ${esc(worst(c).predicted)} was, terwijl het ${esc(worst(c).true)} was. Van de fouten van het CNN had ${fmtPct(share(c))} een zekerheid van 80% of meer. Bij MobileNet, het <em>betere</em> model, was dat ${fmtPct(share(mb))}: minder fouten, maar vaker overtuigd fout. Hieronder een paar testfoto's waar het misging, met wat beide modellen zeiden.`;

  const sec = (id) => state.history[id].seconds;
  const mb2 = (id) => fmtNum(state.onnx.find((o) => o.model === id).onnx_bytes / 1e6, 1);
  $('#reflect-cost').textContent = `Het eigen CNN trainde ${fmtNum(sec('cnn') / 60, 1)} minuten op één videokaart (RTX 2070) en is ${mb2('cnn')} MB groot. MobileNet trainde hier maar ${fmtNum(sec('mobilenet') / 60, 1)} minuten en is ${mb2('mobilenet')} MB, maar leunt op een eerdere training op 1,2 miljoen foto's die veel meer rekenkracht kostte; die kosten zie je hier niet. Elke voorspelling draait daarna op jouw apparaat, zonder datacenter. Kleinere modellen zijn zuiniger, maar hier ook minder goed: dat is een afweging, geen gratis keuze.`;

  const fill = {
    'n-images': fmtNum(Object.values(state.data.trashnet_folders).reduce((a, b) => a + b, 0)),
    'n-train': fmtNum(state.data.split_counts.train.reduce((a, b) => a + b, 0)),
    'n-val': fmtNum(state.data.split_counts.val.reduce((a, b) => a + b, 0)),
    'n-test': fmtNum(state.data.split_counts.test.reduce((a, b) => a + b, 0)),
    'n-trash': fmtNum(state.data.trashnet_folders.trash),
    'n-paper': fmtNum(state.data.trashnet_folders.paper + state.data.trashnet_folders.cardboard),
    'acc-cnn': fmtPct(c.accuracy, 1),
  };
  for (const node of document.querySelectorAll('[data-fill]')) node.textContent = fill[node.dataset.fill];

  const mistakes = state.data.examples.filter((ex) => ex.note);
  $('#mistakes').innerHTML = '';
  for (const ex of mistakes) {
    const card = el('article', { class: 'mistake' });
    const lines = MODELS.map((m) => {
      const p = ex.probabilities[m.id];
      const k = argmax(p);
      const ok = k === ex.label;
      return `<p><span class="swatch" style="--model:${m.color}"></span>${esc(m.name)}: ${esc(state.data.classes[k])} ${fmtPct(p[k])} <span class="badge ${ok ? 'good' : 'bad'}">${ok ? '✓ goed' : '✗ fout'}</span></p>`;
    }).join('');
    card.innerHTML = `<img src="../assets/examples/${esc(ex.file)}" alt="${esc(ex.note)}" loading="lazy">
      <div><p><strong>${esc(ex.note)}</strong></p><p class="muted">Echt: ${esc(state.data.classes[ex.label])}</p>${lines}</div>`;
    const btn = el('button', { type: 'button', class: 'button ghost' }, 'Probeer deze');
    btn.addEventListener('click', () => {
      selectExample(ex);
      $('#probeer').scrollIntoView();
    });
    card.append(btn);
    $('#mistakes').append(card);
  }
}

// ---------- 3 Probeer: kiezen en uploaden ----------

function buildPicker() {
  for (const [selector, list] of [['#thumbs-normal', state.data.examples.filter((e) => !e.note)], ['#thumbs-hard', state.data.examples.filter((e) => e.note)]]) {
    const holder = $(selector);
    for (const ex of list) {
      const b = el('button', { type: 'button', class: 'thumb', 'aria-pressed': 'false', 'data-file': ex.file, title: ex.note || '' },
        `<img src="../assets/examples/${esc(ex.file)}" alt=""><span>${esc(cap(state.data.classes[ex.label]))}</span>`);
      b.addEventListener('click', () => selectExample(ex));
      holder.append(b);
    }
  }
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

async function selectExample(ex) {
  showUploadError('');
  const img = await loadImage(`../assets/examples/${ex.file}`);
  await setSource({ img, name: ex.file, label: ex.label, example: ex });
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
  } catch {
    showUploadError('Deze foto kon niet worden geopend. Probeer een JPG of PNG.');
  } finally {
    URL.revokeObjectURL(url);
    $('#upload').value = '';
  }
}

async function setSource({ img, name, label, example }) {
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const base = canvasOf(w, h);
  base.getContext('2d').drawImage(img, 0, 0, w, h);
  state.source = { base, name, label, example };
  for (const t of document.querySelectorAll('.thumb')) t.setAttribute('aria-pressed', String(t.dataset.file === name && !!example));

  state.base = await predict(base, example);
  drawCanvas($('#try-image'), base);
  $('#try-caption').textContent = example
    ? `Testfoto ${example.source}${example.note ? `: ${example.note.toLowerCase()}` : ''}. Echt: ${state.data.classes[label]}.`
    : `Jouw foto (${name}). Niemand weet hier het juiste antwoord: dat moet jij beoordelen.`;
  renderPredictions($('#try-predictions'), state.base, label);
  renderPixels(state.base.crops.cnn);
  renderFeatureMaps(state.base.tensors.cnn);
  updateTest();
}

// ---------- voorspellen ----------

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
  if (p.kleiner < 1) {
    const s = canvasOf(w, h);
    const sc = s.getContext('2d');
    sc.fillStyle = `rgb(${DARK.join(',')})`;
    sc.fillRect(0, 0, w, h);
    const sw = Math.round(w * p.kleiner);
    const sh = Math.round(h * p.kleiner);
    sc.drawImage(c, Math.floor((w - sw) / 2), Math.floor((h - sh) / 2), sw, sh);
    c = s;
  }
  return c;
}

// Kansen per model. Zonder modellen: de in Python berekende kansen van een voorbeeldfoto.
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
      const out = await state.sessions[m.id].run({ image: new ort.Tensor('float32', tensor, [1, 3, size, size]) });
      result.probs[m.id] = Array.from(out.probabilities.data);
    } else if (example) {
      result.probs[m.id] = example.probabilities[m.id];
    }
  }
  return result;
}

function renderPredictions(container, result, label) {
  const classes = state.data.classes;
  if (!Object.keys(result.probs).length) {
    container.innerHTML = '<div class="pred"><p>Zonder geladen modellen kan deze foto niet beoordeeld worden.</p></div>';
    return;
  }
  container.innerHTML = MODELS.map((m) => {
    const p = result.probs[m.id];
    const k = argmax(p);
    let badge = '';
    if (label !== null && label !== undefined) {
      badge = k === label
        ? '<span class="badge good">✓ klopt</span>'
        : `<span class="badge bad">✗ fout, het is ${esc(classes[label])}</span>`;
    }
    const rows = classes.map((c, i) => ({ label: cap(c), value: p[i], text: fmtPct(p[i]), top: i === k }));
    return `<div class="pred">
      <div class="pred-head"><span class="model-name" style="--model:${m.color}"><span class="swatch"></span>${esc(m.name)} <span class="muted small">${esc(m.detail)}</span></span>${badge}</div>
      <p class="pred-verdict"><strong>${esc(cap(classes[k]))}</strong>, ${fmtPct(p[k])} zeker</p>
      ${hbars(rows, m.color, 1)}
      ${result.live ? '' : '<p class="muted small">Vooraf in Python berekend (modellen niet geladen).</p>'}
    </div>`;
  }).join('');
}

function drawCanvas(target, source) {
  target.width = source.width;
  target.height = source.height;
  target.getContext('2d').drawImage(source, 0, 0);
}

// ---------- 1 Leer: pixels en filters ----------

function rgbToImageData(rgb, size) {
  const img = new ImageData(size, size);
  for (let i = 0, j = 0; i < size * size; i++, j += 3) {
    img.data[i * 4] = rgb[j];
    img.data[i * 4 + 1] = rgb[j + 1];
    img.data[i * 4 + 2] = rgb[j + 2];
    img.data[i * 4 + 3] = 255;
  }
  return img;
}

function renderPixels(crop) {
  const size = state.data.image_size.cnn;
  const canvas = $('#pixel-canvas');
  canvas.getContext('2d').putImageData(rgbToImageData(crop, size), 0, 0);
  state.pixelCrop = crop;
  showPixel(size / 2, size / 2);
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
  const size = state.data.image_size.cnn;
  const i = (py * size + px) * 3;
  const [r, g, b] = [crop[i], crop[i + 1], crop[i + 2]];
  const norm = [r, g, b].map((v, c) => fmtNum((v / 255 - [0.485, 0.456, 0.406][c]) / [0.229, 0.224, 0.225][c], 2));
  $('#pixel-info').innerHTML = `Pixel (${px}, ${py})<br>rood <strong>${r}</strong> · groen <strong>${g}</strong> · blauw <strong>${b}</strong><br><span class="muted">wat het model krijgt: ${norm.join(' · ')}</span>`;
  const zoom = $('#pixel-zoom');
  const zc = zoom.getContext('2d');
  const patch = new ImageData(7, 7);
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      const x = Math.min(size - 1, Math.max(0, px + dx));
      const y = Math.min(size - 1, Math.max(0, py + dy));
      const s = (y * size + x) * 3;
      const d = ((dy + 3) * 7 + dx + 3) * 4;
      patch.data.set([crop[s], crop[s + 1], crop[s + 2], 255], d);
    }
  }
  zc.putImageData(patch, 0, 0);
}

// Echte eerste laag van het CNN: 3x3-convolutie (BatchNorm ingevouwen) + ReLU
function renderFeatureMaps(tensor) {
  const { shape, weight, bias } = state.data.conv1;
  const [O, I, K] = shape;
  const size = state.data.image_size.cnn;
  const n = size * size;
  const grid = $('#feature-maps');
  if (!grid.children.length) {
    for (let o = 0; o < O; o++) grid.append(el('canvas', { width: size, height: size, 'data-tip': `Filter ${o + 1} van ${O}` }));
  }
  const out = new Float32Array(n);
  for (let o = 0; o < O; o++) {
    let max = 0;
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
      }
    }
    const img = new ImageData(size, size);
    for (let p = 0; p < n; p++) {
      const g = max > 0 ? Math.round((out[p] / max) * 255) : 0;
      img.data[p * 4] = g;
      img.data[p * 4 + 1] = g;
      img.data[p * 4 + 2] = g;
      img.data[p * 4 + 3] = 255;
    }
    grid.children[o].getContext('2d').putImageData(img, 0, 0);
  }
}

// ---------- 4 Test: schuifjes ----------

function buildSliders() {
  const holder = $('#sliders');
  for (const s of SLIDERS) {
    const wrap = el('div', { class: 'slider' });
    wrap.innerHTML = `<label for="s-${s.id}"><span>${esc(s.label)}</span><span class="muted mono" id="v-${s.id}">${esc(s.format(s.neutral))}</span></label>
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
  const label = document.querySelector('label[for="upload"]');
  label.setAttribute('aria-disabled', String(!enabled));
  $('#upload').disabled = !enabled;
}

const isNeutral = () => SLIDERS.every((s) => state.params[s.id] === s.neutral);

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
      renderPredictions($('#test-predictions'), result, state.source.label);
    } while (state.testPending);
  } finally {
    state.testBusy = false;
  }
}

// ---------- controle browser tegen Python ----------

// Zelfde formule als synthetic() in src/export_web.py
function syntheticCanvas({ w, h }) {
  const img = new ImageData(w, h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      img.data[i] = Math.floor((x * 255) / (w - 1));
      img.data[i + 1] = Math.floor((y * 255) / (h - 1));
      img.data[i + 2] = (x * 7 + y * 13) % 256;
      img.data[i + 3] = 255;
    }
  }
  const c = canvasOf(w, h);
  c.getContext('2d', { willReadFrequently: true }).putImageData(img, 0, 0);
  return c;
}

const maxAbsDiff = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

async function runParityCheck() {
  const node = $('#parity');
  // 1. Exact dezelfde pixels in browser en Python: test het model en de voorbewerking
  const syn = await predict(syntheticCanvas(state.data.synthetic), null);
  const exactPer = Object.fromEntries(MODELS.map((m) => [m.id, maxAbsDiff(syn.probs[m.id], state.data.synthetic.probabilities[m.id])]));
  const exactDiff = Math.max(...Object.values(exactPer));

  // 2. Echte JPEG-voorbeelden: daar zit ook het openen van de JPEG tussen
  let maxDiff = 0;
  let same = 0;
  let total = 0;
  for (const ex of state.data.examples) {
    const img = await loadImage(`../assets/examples/${ex.file}`);
    const c = canvasOf(img.naturalWidth, img.naturalHeight);
    c.getContext('2d').drawImage(img, 0, 0);
    const res = await predict(c, null);
    for (const m of MODELS) {
      const a = res.probs[m.id];
      const b = ex.probabilities[m.id];
      maxDiff = Math.max(maxDiff, maxAbsDiff(a, b));
      same += argmax(a) === argmax(b) ? 1 : 0;
      total += 1;
    }
  }
  // Float32-afronding verschilt per rekenbibliotheek; 0,1 procentpunt is ruim daarboven
  const ok = same === total && exactDiff < 1e-3;
  node.dataset.state = ok ? 'ok' : 'warn';
  node.innerHTML = `Bij <strong>precies dezelfde pixels</strong> rekent je browser ${ok ? 'vrijwel ' : 'niet '}hetzelfde als Python: het grootste verschil in een percentage is ${fmtNum(exactPer.cnn * 100, 4)} procentpunt (eigen CNN) en ${fmtNum(exactPer.mobilenet * 100, 4)} procentpunt (MobileNet). Die kleine rest komt door afronding van kommagetallen, die net anders verloopt in de browser; bij MobileNet stapelt dat zich op over veel meer lagen. Het verkleinen en uitsnijden gaat tot op de pixel gelijk; dat wordt ook automatisch getest.<br><br>
    Bij de ${state.data.examples.length} voorbeeldfoto's geven beide modellen in ${same} van de ${total} gevallen hetzelfde antwoord als in Python, maar de zekerheid verschilt tot ${fmtNum(maxDiff * 100, 1)} procentpunt. Dat komt doordat je browser een JPEG-bestand net iets anders opent dan Python: veel pixels verschillen 1 of 2 op de 255. Voor het oog onzichtbaar, maar het model merkt het. Ook dat is een les: kleine, onzichtbare verschillen in de invoer kunnen de zekerheid van een model verschuiven.`;
  document.body.dataset.parity = JSON.stringify({ exactPer, synthetic: syn.probs, same, total, maxDiff });
}

// ---------- start ----------

async function loadModels() {
  setStatus('loading', 'Modellen laden (ongeveer 8 MB)…');
  try {
    if (!window.ort) throw new Error('ONNX Runtime Web niet geladen');
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
    ort.env.wasm.numThreads = 1;
    const sessions = await Promise.all(MODELS.map((m) => ort.InferenceSession.create(`../model/${m.id}.onnx`)));
    state.sessions = Object.fromEntries(MODELS.map((m, i) => [m.id, sessions[i]]));
  } catch (err) {
    console.error(err);
    setStatus('fallback', 'De modellen konden niet laden (offline of geblokkeerd). Je ziet de vooraf in Python berekende uitkomsten voor de voorbeeldfoto\'s; eigen foto\'s en de schuifjes werken niet.');
    $('#parity').textContent = 'Niet gecontroleerd: de modellen zijn niet geladen.';
    setLive(false);
    return;
  }
  setStatus('ready', 'Modellen geladen. Alles wordt in je browser berekend.');
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
    setStatus('error', 'De gegevens konden niet worden geladen. Open de pagina via een webserver (niet door te dubbelklikken), bijvoorbeeld met: python -m http.server');
    return;
  }

  renderDataset();
  segmented($('#curve-model'), MODELS.map((m) => ({ id: m.id, label: m.name })), state.curveModel, (id) => { state.curveModel = id; renderCurves(); });
  segmented($('#curve-metric'), [{ id: 'accuracy', label: 'Goed (accuracy)' }, { id: 'loss', label: 'Fout-score (loss)' }], state.curveMetric, (id) => { state.curveMetric = id; renderCurves(); });
  renderCurves();
  renderStatTiles();
  segmented($('#cm-model'), MODELS.map((m) => ({ id: m.id, label: m.name })), state.cmModel, (id) => { state.cmModel = id; renderConfusion(); });
  renderConfusion();
  renderClassTable();
  renderVariations();
  renderReflection();
  buildPicker();
  buildSliders();
  setLive(false);
  $('#pixel-canvas').addEventListener('pointermove', pixelHover);

  // Een duidelijk voorwerp als eerste indruk (een glazen fles), geen close-up van karton
  const first = state.data.examples.find((e) => !e.note && state.data.classes[e.label] === 'glas') || state.data.examples[0];
  await selectExample(first);
  loadModels();
}

init();
