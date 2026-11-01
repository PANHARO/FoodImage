import * as ort from "./vendor/ort.wasm.min.mjs";
import {prepare} from "./preprocess.mjs";

// One CPU/WASM session per page visit, isolated from the UI thread.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.proxy = false;
let sessionPromise;
async function loadModel() {
  postMessage({type: "status", message: "Loading model…"});
  const response = await fetch(new URL("./model/food.onnx", import.meta.url));
  if (!response.ok) throw new Error("The model could not be downloaded.");
  const data = await response.arrayBuffer();
  postMessage({type: "status", message: "Preparing model…"});
  const session = await ort.InferenceSession.create(data, {executionProviders: ["wasm"]});
  postMessage({type: "ready"});
  return session;
}
function session() {
  if (!sessionPromise) sessionPromise = loadModel().catch(error => {sessionPromise = undefined; throw error;});
  return sessionPromise;
}
let queue = Promise.resolve();
onmessage = ({data}) => {
  if (data.type === "init") {
    session().catch(error => postMessage({type: "error", message: error.message}));
    return;
  }
  queue = queue.then(async () => {
    try {
      const model = await session();
      const start = performance.now();
      const tensor = new ort.Tensor("float32", prepare(new Uint8Array(data.rgb), data.width, data.height), [1, 3, 224, 224]);
      const result = await model.run({image: tensor});
      const scores = Array.from(result.probabilities.data);
      tensor.dispose();
      result.probabilities.dispose();
      postMessage({type: "result", id: data.id, scores, milliseconds: performance.now() - start});
    } catch (error) { postMessage({type: "error", id: data.id, message: error.message || "Inference failed."}); }
  });
};
