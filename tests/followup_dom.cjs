// Run the delivered script against synthetic PNG bytes. This is not a browser layout test.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {webcrypto}=require('node:crypto');
const page=fs.readFileSync(process.argv[2],'utf8');
const dataText=page.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const script=page.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(dataText),elements={};
for(const match of page.matchAll(/id="([^"]+)"/g))elements[match[1]]={value:'',textContent:'',style:{},disabled:false,checked:false,removeAttribute(k){delete this[k];}};
elements.data.textContent=dataText;elements.zoom.value='1';elements.reviewer.value='合成复核人';
let serial=0,saved,unload;const blobs=new Map();
class FakeImage{async decode(){const blob=blobs.get(this.src),bytes=Buffer.from(await blob.arrayBuffer());this.naturalWidth=bytes.readUInt32BE(16);this.naturalHeight=bytes.readUInt32BE(20);}}
const context={document:{getElementById:id=>elements[id],createElement:()=>({click(){saved=blobs.get(this.href);}})},
  window:{addEventListener:(name,fn)=>{if(name==='beforeunload')unload=fn;}},crypto:webcrypto,Image:FakeImage,Blob,
  URL:{createObjectURL:blob=>{const key='blob:'+(++serial);blobs.set(key,blob);return key;},revokeObjectURL:key=>blobs.delete(key)},setTimeout:fn=>fn()};
vm.runInNewContext(script,context);
const assert=(ok,message)=>{if(!ok)throw Error(message);};
const msg=()=>elements.message.textContent;
const makeFile=(name,bytes)=>{const blob=new Blob([bytes]);blob.name=name;return blob;};
const files=Object.values(data.images).map(im=>makeFile(im.name,fs.readFileSync(path.join(process.argv[3],im.name))));
const choose=async values=>elements.files.onchange({target:{files:values,value:'selected'}});
const restore=async value=>elements.restore.onchange({target:{files:[{text:async()=>JSON.stringify(value)}],value:'selected'}});
const download=async()=>{elements.download.onclick();return JSON.parse(await saved.text());};
(async()=>{
  elements.save.onclick();assert(msg().includes('加载'),'Unverified images accepted');
  const unrelated={name:'not-requested.png',arrayBuffer(){throw Error('Read unrelated image');}};
  await choose([files[0],unrelated]);assert(elements.save.disabled,'Partial pair enabled');
  await choose([files[1]]);assert(!elements.save.disabled,'Correct hashes not accepted');
  assert(elements['left-image'].style.width==='1024px','Original pixel view is not default');
  elements.zoom.value='2';elements.zoom.onchange();assert(elements['right-image'].style.width==='2048px','Zoom failed');
  elements.status.value='uncertain';elements.reason.value='合成图缺少固定地物，无法确定关系';elements.reason.oninput();
  elements.next.onclick();assert(msg().includes('先保存'),'Dirty navigation discarded work');
  elements.save.onclick();assert(msg().includes('勾选'),'Missing viewing declaration accepted');
  elements.viewed.checked=true;elements.save.onclick();
  let warned=false;unload({preventDefault(){warned=true;}});assert(warned,'Unsaved download warning missing');
  const first=await download();assert(first.decisions.length===1,'Partial progress export failed');
  await restore(first);elements.reviewer.value='另一人';
  assert(JSON.stringify((await download()).decisions)===JSON.stringify(first.decisions),'Recovery changed original provenance');
  const cases=[['reviewed_at','2026-02-30T08:00:00.000Z'],['reviewed_at','2026-09-14T08:00:00'],['reason',''],['left','bad'],['viewed_originals',false],['image_sha256',{}]];
  for(const [key,value] of cases){const bad=structuredClone(first);bad.decisions[0][key]=value;await restore(bad);assert(msg().includes('恢复失败'),'Bad field accepted: '+key);assert(JSON.stringify((await download()).decisions)===JSON.stringify(first.decisions),'Failed import destroyed work');}
  for(const bad of [{...first,request_sha256:'wrong'},{...first,kind:'scene-decisions'},{...first,decisions:[...first.decisions,...first.decisions]}]){
    await restore(bad);assert(msg().includes('恢复失败'),'Stale or duplicate record accepted');
  }
  await choose([makeFile(files[0].name,Buffer.alloc(files[0].size))]);assert(elements.save.disabled,'Wrong hash permitted edit');
  assert(elements['left-image'].src===undefined,'Stale image remained visible after mismatch');
  assert(JSON.stringify((await download()).decisions)===JSON.stringify(first.decisions),'Image mismatch destroyed saved work');
  await choose([files[0],files[0]]);assert(elements.save.disabled,'Ambiguous filename accepted');
  await choose([files[0],files[2]]);assert(!elements.save.disabled,'Retry did not recover');
  elements.reason.value='未保存的变更';elements.reason.oninput();await restore(first);assert(msg().includes('先保存'),'Dirty import accepted');
  elements.discard.onclick();assert(elements.reason.value===first.decisions[0].reason,'Discard lost saved answer');
  console.log(JSON.stringify(await download()));
})().catch(error=>{console.error(error);process.exitCode=1;});
