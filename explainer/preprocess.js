// Voorbewerking, gelijk aan src/preprocess.py (eval_transform) en src/variations.py.
//
// Pure functies op pixelarrays, zonder DOM, zodat tests/js_preprocess.mjs ze in Node
// kan vergelijken met Python. Het schalen volgt Pillow's bilineaire resize tot op de
// vaste-kommaberekening, zodat de browser het model dezelfde getallen geeft als PyTorch.

export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];
export const DARK = [40, 40, 40];
export const REFERENCE_WIDTH = 512;

const PRECISION_BITS = 32 - 8 - 2;

// RGBA (ImageData.data) naar RGB
export function toRGB(rgba, w, h) {
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h * 4; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return rgb;
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc, bilineair filter
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filterscale;
  const ss = 1 / filterscale;
  const bounds = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    const k = [];
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      const w = t < 1 ? 1 - t : 0;
      k.push(w);
      ww += w;
    }
    const kk = k.map((w) => {
      const v = (ww !== 0 ? w / ww : w) * (1 << PRECISION_BITS);
      return v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5);
    });
    bounds.push({ xmin, kk });
  }
  return bounds;
}

function clip8(acc) {
  const v = Math.floor(acc / (1 << PRECISION_BITS));
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function resampleHorizontal(src, w, h, outW) {
  const out = new Uint8Array(outW * h * 3);
  const coeffs = coefficients(w, outW);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < outW; xx++) {
      const { xmin, kk } = coeffs[xx];
      for (let c = 0; c < 3; c++) {
        let acc = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < kk.length; x++) acc += src[(y * w + x + xmin) * 3 + c] * kk[x];
        out[(y * outW + xx) * 3 + c] = clip8(acc);
      }
    }
  }
  return out;
}

function resampleVertical(src, w, h, outH) {
  const out = new Uint8Array(w * outH * 3);
  const coeffs = coefficients(h, outH);
  for (let yy = 0; yy < outH; yy++) {
    const { xmin: ymin, kk } = coeffs[yy];
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < 3; c++) {
        let acc = 1 << (PRECISION_BITS - 1);
        for (let y = 0; y < kk.length; y++) acc += src[((y + ymin) * w + x) * 3 + c] * kk[y];
        out[(yy * w + x) * 3 + c] = clip8(acc);
      }
    }
  }
  return out;
}

// Pillow Image.resize(..., BILINEAR): eerst horizontaal, dan verticaal, elk alleen als nodig
export function resize(rgb, w, h, outW, outH) {
  let data = rgb;
  if (outW !== w) data = resampleHorizontal(data, w, h, outW);
  if (outH !== h) data = resampleVertical(data, outW, h, outH);
  return data;
}

// torchvision Resize(size): korte zijde wordt size, lange zijde int(size * lang / kort)
export function resizedSize(w, h, size) {
  return w <= h ? [size, Math.trunc((size * h) / w)] : [Math.trunc((size * w) / h), size];
}

// Python round(): bij precies .5 naar het even getal
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function centerCrop(rgb, w, h, size) {
  const top = roundHalfEven((h - size) / 2);
  const left = roundHalfEven((w - size) / 2);
  const out = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    out.set(rgb.subarray(((y + top) * w + left) * 3, ((y + top) * w + left + size) * 3), y * size * 3);
  }
  return out;
}

// RGB-uint8 (HWC) naar genormaliseerde float32 (CHW): ToTensor + Normalize
export function toTensor(rgb, size) {
  const n = size * size;
  const t = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) t[c * n + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
  }
  return t;
}

// Volledige eval_transform: geeft het vierkante RGB-beeld (om te tonen) en de tensor
export function preprocess(rgb, w, h, size) {
  const [rw, rh] = resizedSize(w, h, size);
  const crop = centerCrop(resize(rgb, w, h, rw, rh), rw, rh, size);
  return { crop, tensor: toTensor(crop, size) };
}

// ImageEnhance.Brightness: mengen met zwart, dus elke waarde maal factor
export function brightness(rgba, factor) {
  for (let i = 0; i < rgba.length; i += 4) {
    for (let c = 0; c < 3; c++) rgba[i + c] = Math.min(255, Math.max(0, Math.round(rgba[i + c] * factor)));
  }
}

// ImageFilter.BoxBlur(radius): gemiddelde over 2r+1 pixels, randpixels herhaald,
// breuk van de straal telt mee voor de buitenste pixels
function blurLine(src, dst, n, stride, offset, radius) {
  const ri = Math.floor(radius);
  const frac = radius - ri;
  const norm = 2 * radius + 1;
  const at = (i) => src[offset + Math.min(n - 1, Math.max(0, i)) * stride];
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let k = -ri; k <= ri; k++) acc += at(i + k);
    acc += frac * (at(i - ri - 1) + at(i + ri + 1));
    dst[offset + i * stride] = Math.round(acc / norm);
  }
}

export function boxBlur(rgba, w, h, radius) {
  if (radius <= 0) return;
  const tmp = new Uint8ClampedArray(rgba.length);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < h; y++) blurLine(rgba, tmp, w, 4, y * w * 4 + c, radius);
    for (let x = 0; x < w; x++) blurLine(tmp, rgba, h, w * 4, x * 4 + c, radius);
  }
}
