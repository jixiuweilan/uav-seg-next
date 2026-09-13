// Exercise the actual generated page script with a minimal DOM and Blob API.
// This verifies import/export logic; it is not a browser layout test.
const fs = require('node:fs');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');
const page = fs.readFileSync(process.argv[2], 'utf8');
const dataText = page.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const script = page.match(/<script>\n([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(dataText);
const cards = data.pairs.map(() => {
  const fields = Object.fromEntries(['status','reason','evidence'].map(k=>['.'+k,{value:''}]));
  return {fields,querySelector: selector=>fields[selector]};
});
const elements = {
  data:{textContent:dataText},message:{textContent:''},reviewer:{value:'Synthetic Reviewer'},
  export:{},import:{},
  'manual-left':{value:''},'manual-right':{value:''},'manual-status':{value:'rejected'},
  'manual-evidence':{value:'source_metadata'},'manual-reason':{value:'Synthetic source records show different flights.'},
  'manual-add':{},'manual-remove':{},'manual-list':{textContent:''}
};
let saved;
const context = {
  document:{getElementById:id=>elements[id],querySelectorAll:()=>cards,createElement:()=>({click(){}})},
  Blob, URL:{createObjectURL:blob=>{saved=blob;return 'blob:test';},revokeObjectURL(){}},
  crypto:webcrypto,TextEncoder,setTimeout: fn=>fn()
};
vm.runInNewContext(script,context);
(async()=>{
  cards[0].fields['.status'].value='confirmed';
  cards[0].fields['.reason'].value='Synthetic original images share the same landmarks.';
  cards[0].fields['.evidence'].value='full_resolution_images';
  let manualAdded=false;
  for(let i=0;i<data.training_ids.length&&!manualAdded;i++)for(let j=i+1;j<data.training_ids.length&&!manualAdded;j++){
    elements['manual-left'].value=data.training_ids[i];
    elements['manual-right'].value=data.training_ids[j];
    await elements['manual-add'].onclick();
    manualAdded=elements.message.textContent.startsWith('Manual relation added.');
  }
  if(!manualAdded)throw Error('Could not find an unscreened pair for manual entry');
  elements.export.onclick();
  const first=JSON.parse(await saved.text());
  if(first.decisions.length!==2||!first.decisions.some(row=>row.origin==='manual'))throw Error('Manual decision was not exported');
  await elements.import.onchange({target:{files:[{text:async()=>JSON.stringify(first)}]}});
  elements.reviewer.value='Different Reviewer';
  elements.export.onclick();
  const second=JSON.parse(await saved.text());
  if(JSON.stringify(first.decisions)!==JSON.stringify(second.decisions))throw Error('Unchanged imported decisions lost provenance');
  const noTimezone=JSON.parse(JSON.stringify(first));
  noTimezone.decisions[0].reviewed_at='2026-09-13T17:30:00';
  await elements.import.onchange({target:{files:[{text:async()=>JSON.stringify(noTimezone)}]}});
  if(!elements.message.textContent.includes('timezone'))throw Error('Timezone-less timestamp was accepted');
  const bad={...first,manifest_sha256:'wrong'};
  await elements.import.onchange({target:{files:[{text:async()=>JSON.stringify(bad)}]}});
  if(!elements.message.textContent.includes('mismatch'))throw Error('Stale identity was accepted');
  console.log(JSON.stringify(second));
})().catch(error=>{console.error(error);process.exitCode=1;});
