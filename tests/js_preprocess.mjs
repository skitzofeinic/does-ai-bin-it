// Hulpscript voor test_preprocess_js.py: leest {rgb (base64), w, h, size} van stdin,
// draait explainer/preprocess.js en schrijft {crop (base64), tensor} naar stdout.
import { preprocess } from '../explainer/preprocess.js';

let input = '';
for await (const chunk of process.stdin) input += chunk;
const { rgb, w, h, size } = JSON.parse(input);
const { crop, tensor } = preprocess(new Uint8Array(Buffer.from(rgb, 'base64')), w, h, size);
process.stdout.write(JSON.stringify({
  crop: Buffer.from(crop).toString('base64'),
  tensor: Array.from(tensor),
}));
