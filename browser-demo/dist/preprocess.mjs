// Match Pillow's separable, antialiased 8-bit bilinear resize.
const UNIT = 2 ** 22;
function axis(source, target) {
  const scale = source / target, radius = Math.max(1, scale);
  return Array.from({length: target}, (_, output) => {
    const center = (output + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - radius + 0.5));
    const end = Math.min(source, Math.trunc(center + radius + 0.5));
    const values = Array.from({length: end - start}, (_, i) =>
      Math.max(0, 1 - Math.abs((start + i + 0.5 - center) / radius)));
    const total = values.reduce((sum, value) => sum + value, 0);
    return {start, weights: values.map(value => Math.round(value / total * UNIT))};
  });
}
export function resizeRGB(rgb, width, height, size = 224) {
  if (width === size && height === size) return new Uint8Array(rgb);
  const horizontal = axis(width, size), vertical = axis(height, size);
  const intermediate = new Uint8Array(size * height * 3);
  for (let y = 0; y < height; y++) for (let x = 0; x < size; x++) {
    const {start, weights} = horizontal[x];
    for (let c = 0; c < 3; c++) {
      let sum = UNIT / 2;
      weights.forEach((weight, i) => { sum += rgb[(y * width + start + i) * 3 + c] * weight; });
      intermediate[(y * size + x) * 3 + c] = Math.min(255, Math.max(0, Math.floor(sum / UNIT)));
    }
  }
  const resized = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const {start, weights} = vertical[y];
    for (let c = 0; c < 3; c++) {
      let sum = UNIT / 2;
      weights.forEach((weight, i) => { sum += intermediate[((start + i) * size + x) * 3 + c] * weight; });
      resized[(y * size + x) * 3 + c] = Math.min(255, Math.max(0, Math.floor(sum / UNIT)));
    }
  }
  return resized;
}
export function prepare(rgb, width, height) {
  const image = resizeRGB(rgb, width, height);
  const output = new Float32Array(3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406].map(Math.fround);
  const std = [0.229, 0.224, 0.225].map(Math.fround);
  for (let c = 0; c < 3; c++) for (let i = 0; i < 224 * 224; i++)
    output[c * 224 * 224 + i] = Math.fround(Math.fround(Math.fround(image[i * 3 + c] / 255) - mean[c]) / std[c]);
  return output;
}
