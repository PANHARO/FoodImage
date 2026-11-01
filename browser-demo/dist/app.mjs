const NAMES = ["Adobo","Amok trey","Banh mi","Hainanese chicken rice","Laksa","Laphet thoke","Nasi goreng","Pad Thai","Pho","Satay"];
const $ = id => document.getElementById(id);
const worker = new Worker(new URL("./inference.mjs", import.meta.url), {type:"module"});
const panel = document.querySelector(".output-panel");
let requestId = 0, objectURL;
NAMES.forEach(name => {const chip=document.createElement("span");chip.textContent=name;$("dish-list").append(chip);});

function modelStatus(text, state="") {
  $("model-status").className = "status " + state;
  $("status-text").textContent = text;
}
function clearScores() {
  $("bars").replaceChildren();
  $("bars").hidden=true;
  $("confidence").hidden=true;
  $("timing").hidden=true;
}
function message(title, text, kicker="THE PREDICTION") {
  $("dish").textContent=title;
  $("result-message").textContent=text;
  $("result-kicker").textContent=kicker;
}
function failure(text) {
  panel.classList.remove("busy");
  panel.classList.add("error");
  $("result").setAttribute("aria-busy","false");
  clearScores();
  document.querySelector(".idle-mark").hidden=true;
  message("Let’s try that again.", text);
}
worker.onmessage = ({data}) => {
  if(data.type==="status"){modelStatus(data.message);return;}
  if(data.type==="ready"){modelStatus("Model ready","ready");return;}
  if(data.type==="error"){
    if(data.id===undefined){modelStatus("Model unavailable","failed");}
    if(data.id===undefined||data.id===requestId) failure("The photo could not be analyzed. Choose a photo to try again, or refresh if the model is unavailable.");
    console.error(data.message);
    return;
  }
  if(data.type!=="result"||data.id!==requestId)return;
  panel.classList.remove("busy","error");
  $("result").setAttribute("aria-busy","false");
  document.querySelector(".idle-mark").hidden=true;
  clearScores();
  const predictions=data.scores.map((score,index)=>({name:NAMES[index],score})).sort((a,b)=>b.score-a.score).slice(0,3);
  if(predictions[0].score<0.5){
    message("Not sure.", "This model only knows these 10 dishes: " + NAMES.join(", "), "A LITTLE OUTSIDE MY MENU");
    return;
  }
  message(predictions[0].name, "", "HERE’S MY BEST GUESS");
  $("confidence-value").textContent=(predictions[0].score*100).toFixed(1)+"%";
  $("confidence").hidden=false;
  for(const prediction of predictions){
    const row=document.createElement("div");row.className="bar-row";
    const label=document.createElement("div");label.className="bar-label";
    const name=document.createElement("span");name.textContent=prediction.name;
    const score=document.createElement("span");score.textContent=(prediction.score*100).toFixed(1)+"%";
    label.append(name,score);
    const track=document.createElement("div");track.className="bar-track";
    track.setAttribute("role","meter");track.setAttribute("aria-label",prediction.name+" confidence");
    track.setAttribute("aria-valuemin","0");track.setAttribute("aria-valuemax","100");
    track.setAttribute("aria-valuenow",String(prediction.score*100));
    const fill=document.createElement("div");fill.className="bar-fill";fill.style.width=(prediction.score*100)+"%";
    track.append(fill);row.append(label,track);$("bars").append(row);
  }
  $("bars").hidden=false;
  $("timing").textContent="Analyzed on your device · "+(data.milliseconds/1000).toFixed(2)+"s";
  $("timing").hidden=false;
};
worker.onerror=()=>{modelStatus("Model unavailable","failed");failure("The model could not start. Refresh this page to try again in a recent browser.");};

async function analyze(blob, selectedExample) {
  const id=++requestId;
  if(blob.size>20*1024*1024){failure("Please choose a photo smaller than 20 MB.");return;}
  panel.classList.add("busy");panel.classList.remove("error");
  $("result").setAttribute("aria-busy","true");
  clearScores();
  document.querySelector(".idle-mark").hidden=false;
  message("Taking a closer look…","Your photo is being analyzed on your device.","ONE MOMENT");
  document.querySelectorAll(".example").forEach(button=>button.classList.toggle("active",button.dataset.dish===selectedExample));
  try {
    const bitmap=await createImageBitmap(blob,{imageOrientation:"from-image",premultiplyAlpha:"none",colorSpaceConversion:"none"});
    if(id!==requestId){bitmap.close();return;}
    if(bitmap.width*bitmap.height>24000000){bitmap.close();throw new Error("Please choose a photo under 24 megapixels.");}
    const canvas=document.createElement("canvas");canvas.width=bitmap.width;canvas.height=bitmap.height;
    const context=canvas.getContext("2d",{willReadFrequently:true});
    context.drawImage(bitmap,0,0);bitmap.close();
    const rgba=context.getImageData(0,0,canvas.width,canvas.height).data;
    const rgb=new Uint8Array(canvas.width*canvas.height*3);
    for(let i=0,j=0;i<rgba.length;i+=4){rgb[j++]=rgba[i];rgb[j++]=rgba[i+1];rgb[j++]=rgba[i+2];}
    if(objectURL)URL.revokeObjectURL(objectURL);
    objectURL=URL.createObjectURL(blob);$("preview").src=objectURL;
    $("preview").hidden=false;$("drop-prompt").hidden=true;$("change-photo").hidden=false;
    $("upload").classList.add("has-image");$("clear").disabled=false;
    worker.postMessage({type:"predict",id,rgb:rgb.buffer,width:canvas.width,height:canvas.height},[rgb.buffer]);
  }catch(error){if(id===requestId)failure(error.message||"Please choose a JPG, PNG or WebP photo.");}
}
$("upload").addEventListener("click",()=>$("photo").click());
$("photo").addEventListener("change",event=>{
  const file=event.target.files[0];if(file)analyze(file);event.target.value="";
});
$("clear").addEventListener("click",()=>{
  requestId++;
  if(objectURL){URL.revokeObjectURL(objectURL);objectURL=undefined;}
  $("preview").hidden=true;$("preview").removeAttribute("src");
  $("drop-prompt").hidden=false;$("change-photo").hidden=true;
  $("upload").classList.remove("has-image");$("clear").disabled=true;
  panel.classList.remove("busy","error");$("result").setAttribute("aria-busy","false");
  document.querySelector(".idle-mark").hidden=false;clearScores();
  message("Ready when you are.","Add a photo or choose an example below. Let’s see what’s cooking.","A LITTLE FOOD FOR THOUGHT");
  document.querySelectorAll(".example").forEach(button=>button.classList.remove("active"));
});
for(const event of ["dragenter","dragover"])$("upload").addEventListener(event,e=>{e.preventDefault();$("upload").classList.add("dragging");});
for(const event of ["dragleave","drop"])$("upload").addEventListener(event,e=>{e.preventDefault();$("upload").classList.remove("dragging");});
$("upload").addEventListener("drop",e=>{const file=e.dataTransfer.files[0];if(file)analyze(file);});
document.querySelectorAll(".example").forEach(button=>button.addEventListener("click",async()=>{
  const name=button.dataset.dish;
  try{const response=await fetch(new URL("./examples/"+name+".jpg",import.meta.url));if(!response.ok)throw new Error();await analyze(await response.blob(),name);}
  catch{failure("This example could not be loaded. Try uploading a photo.");}
}));
worker.postMessage({type:"init"});
