"""Human-declared scene decisions, confirmed components and local review pages."""

from datetime import datetime
from io import BytesIO
import base64
import html
from pathlib import Path

import numpy as np
from PIL import Image

from .common import AuditError, canonical, new_output, sha256
from .scenes import POSES, pair_id, producer, split_membership, training_rows, transformed
from .scenes import validate_screen, verified_image


def empty_decisions(document, screen):
    return {"schema_version": 1, "kind": "scene-decisions",
            "manifest_sha256": document["manifest_sha256"],
            "screen_sha256": screen["screen_sha256"], "decisions": []}


def group_report(document, screen, decisions=None):
    pairs = validate_screen(screen, document)
    decisions = empty_decisions(document, screen) if decisions is None else decisions
    template = empty_decisions(document, screen)
    if not isinstance(decisions, dict) or any(decisions.get(k) != v for k, v in template.items()
                                            if k != "decisions"):
        raise AuditError("decision schema or manifest/screen identity mismatch")
    records = decisions.get("decisions")
    if not isinstance(records, list):
        raise AuditError("decisions must be a list")
    ids = training_rows(document)
    membership = split_membership(document, ids)
    parent = {s: s for s in ids}

    def root(s):
        while s != parent[s]:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    parsed = {}
    for record in records:
        if not isinstance(record, dict) or record.get("left") not in ids or record.get("right") not in ids:
            raise AuditError("decision references an unknown training ID")
        key = pair_id(record["left"], record["right"])
        if record.get("pair_id") != key or key in parsed:
            raise AuditError("invalid or duplicate decision pair ID")
        if record.get("status") not in ("confirmed", "rejected", "uncertain"):
            raise AuditError("decision status must be confirmed, rejected or uncertain")
        for field in ("reviewer", "reason", "reviewed_at"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise AuditError(f"decision requires {field}")
        if record.get("evidence") not in ("full_resolution_images", "source_metadata"):
            raise AuditError("decision requires a declared evidence type")
        try:
            timestamp = datetime.fromisoformat(record["reviewed_at"])
            if timestamp.tzinfo is None:
                raise ValueError("timezone missing")
        except ValueError as exc:
            raise AuditError("reviewed_at must be an ISO timestamp with timezone") from exc
        if record.get("origin") != ("screen" if key in pairs else "manual"):
            raise AuditError("unscreened relations require explicit manual origin")
        parsed[key] = record
        if record["status"] == "confirmed":
            a, b = sorted((root(record["left"]), root(record["right"])))
            parent[b] = a

    members = {}
    for sample_id in ids:
        members.setdefault(root(sample_id), []).append(sample_id)
    groups = []
    for values in sorted(members.values()):
        if len(values) < 2:
            continue
        member_set = set(values)
        edges = sorted(key for key, r in parsed.items() if r["status"] == "confirmed"
                       and r["left"] in member_set and r["right"] in member_set)
        groups.append({"group_id": sha256(canonical([document["manifest_sha256"], values])),
                       "members": values, "confirmed_edges": edges})
    crossing = [g["group_id"] for g in groups if membership and
                len({membership[s] for s in g["members"]}) > 1]
    contradictions = sorted(key for key, r in parsed.items() if r["status"] == "rejected"
                            and root(r["left"]) == root(r["right"]))
    pending = sorted(set(pairs) - set(parsed))
    uncertain = sorted(key for key, r in parsed.items() if r["status"] == "uncertain")
    oversized = [g["group_id"] for g in groups if len(g["members"]) > 20]
    omitted = sum(screen["screen"]["omitted_by_partition"].values())
    status = "reviewed_cues_only"
    if pending or uncertain or omitted or oversized:
        status = "review_pending"
    if crossing:
        status = "known_split_conflict"
    if contradictions:
        status = "inconsistent_review"
    counts = {queue: {state: 0 for state in ("confirmed", "rejected", "uncertain", "pending")}
              for queue in ("candidate", "control", "manual")}
    for key, pair in pairs.items():
        counts[pair["queue"]][parsed.get(key, {}).get("status", "pending")] += 1
    for key, record in parsed.items():
        if key not in pairs:
            counts["manual"][record["status"]] += 1
    payload = {"schema_version": 1, "kind": "scene-group-report", "status": status,
               "manifest_sha256": document["manifest_sha256"], "screen_sha256": screen["screen_sha256"],
               "decisions_sha256": sha256(canonical(decisions)), "groups": groups,
               "ungrouped_ids": [s for s in ids if len(members[root(s)]) == 1],
               "cross_split_group_ids": crossing, "contradictory_rejections": contradictions,
               "unreviewed_pair_ids": pending, "uncertain_pair_ids": uncertain,
               "groups_over_20_members": oversized, "omitted_candidates": omitted,
               "review_counts": counts, "split_check_available": bool(membership),
               "scene_independence_certified": False,
               "reviewer_identity_authenticated": False}
    return {"report": payload, "report_sha256": sha256(canonical(payload)), "producer": producer()}


def jpeg_uri(image):
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def review_page(paths, document, screen, output, protected):
    pairs = validate_screen(screen, document)
    rows = training_rows(document)
    needed = sorted({row[side] for row in pairs.values() for side in ("left", "right")})
    previews, originals, thumbnails = {}, {}, {}
    # The artifact contains local file links; write it only to ignored local storage.
    if ".local" not in Path(output).resolve().parts:
        raise AuditError("review HTML with local original-image links must be under .local/")
    with new_output(output, protected) as temporary:
        for sample_id in needed:
            with verified_image(paths, rows[sample_id]) as image:
                thumb = image.resize((256, 256), Image.Resampling.LANCZOS)
            thumbnails[sample_id] = thumb
            previews[sample_id] = jpeg_uri(thumb)
            originals[sample_id] = (paths.folders["train_images"] / (sample_id + ".png")).resolve().as_uri()
        cards = []
        for index, row in enumerate(pairs.values()):
            left, right = row["left"], row["right"]
            aligned = Image.fromarray(transformed(np.asarray(thumbnails[right]), POSES.index(row["pose"])))
            images = "".join(
                f'<a href="{html.escape(originals[s])}" target="_blank" rel="noopener">'
                f'<img src="{previews[s]}" alt="{html.escape(s)}"><span>{html.escape(s)} — open original</span></a>'
                for s in (left, right))
            cross = "cross split" if row["cross_split"] else "within split / unassigned"
            cards.append(f'''<article data-index="{index}">
<h2>{index + 1}. {html.escape(left)} / {html.escape(right)}</h2>
<p>{row['queue']} · {cross} · RMS {row['rms']} · hash distance {row['hamming']} · low texture {row['low_texture']}</p>
<div class="images">{images}</div>
<details><summary>Best aligned right preview: {row['pose']}</summary><img src="{jpeg_uri(aligned)}" alt="Aligned right preview"></details>
<label>Conclusion <select class="status"><option value="">Unreviewed</option><option value="confirmed">Confirmed: same scene / duplicate</option><option value="rejected">Rejected: different scenes</option><option value="uncertain">Uncertain</option></select></label>
<label>Evidence <select class="evidence"><option value="">Choose evidence</option><option value="full_resolution_images">Full-resolution originals</option><option value="source_metadata">Source / flight metadata</option></select></label>
<label>Reason / source reference <textarea class="reason" rows="2"></textarea></label>
</article>''')
        data = {"base": empty_decisions(document, screen), "pairs": list(pairs.values())}
        encoded = canonical(data).decode().replace("<", "\\u003c")
        page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training scene review</title>
<style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 16px;color:#172033;background:#f4f6f8}header,article{background:white;padding:20px;border:1px solid #d8dfe8;border-radius:8px;margin:16px 0}header{position:sticky;top:0;z-index:1}h1{font-size:24px}h2{font-size:18px}.images{display:flex;gap:16px;flex-wrap:wrap}img{width:256px;height:256px}a span{display:block}label{display:block;margin-top:12px}textarea{width:95%}button,input,select{font:inherit;padding:5px}#message{font-weight:bold}</style>
<header><h1>Training scene review</h1>
<p>Candidates are cues. Compare full-resolution originals before recording a decision. No files are uploaded; this page does not modify the dataset.</p>
<label>Reviewer <input id="reviewer" placeholder="Your name or reviewer ID"></label>
<button id="export">Download decisions</button> <label>Resume decisions <input type="file" id="import" accept=".json"></label>
<p id="message" role="status">Decisions are not saved until downloaded.</p></header>
''' + "\n".join(cards) + '<script type="application/json" id="data">' + encoded + '''</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const cards=[...document.querySelectorAll('article')];
const message=document.getElementById('message');
let prior=new Map(), manual=[];
document.getElementById('export').onclick=()=>{
 try {
  const reviewer=document.getElementById('reviewer').value.trim();
  const decisions=[...manual];
  cards.forEach((card,i)=>{
   const status=card.querySelector('.status').value;
   if(!status)return;
   const reason=card.querySelector('.reason').value.trim(),evidence=card.querySelector('.evidence').value;
   if(!reviewer||!reason||!evidence)throw Error('Each decision needs reviewer, evidence and reason.');
   const pair=data.pairs[i],old=prior.get(pair.pair_id);
   const unchanged=old&&old.status===status&&old.reason===reason&&old.evidence===evidence;
   decisions.push({pair_id:pair.pair_id,left:pair.left,right:pair.right,status,reason,evidence,origin:'screen',reviewer:unchanged?old.reviewer:reviewer,reviewed_at:unchanged?old.reviewed_at:new Date().toISOString()});
  });
  const blob=new Blob([JSON.stringify({...data.base,decisions},null,2)+'\\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='scene-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  message.textContent=`Downloaded ${decisions.length} decisions. Keep each revision as a separate file.`;
 }catch(error){message.textContent=error.message;}
};
document.getElementById('import').onchange=async event=>{
 try{
  const value=JSON.parse(await event.target.files[0].text());
  for(const key of ['schema_version','kind','manifest_sha256','screen_sha256'])if(value[key]!==data.base[key])throw Error('Decision identity mismatch.');
  if(!Array.isArray(value.decisions))throw Error('Invalid decisions.');
  const next=new Map(),extras=[],known=new Map(data.pairs.map(p=>[p.pair_id,p]));
  for(const row of value.decisions){
   if(next.has(row.pair_id)||!['confirmed','rejected','uncertain'].includes(row.status)||!row.reason||!row.reviewer||!row.reviewed_at||!['full_resolution_images','source_metadata'].includes(row.evidence))throw Error('Invalid or duplicate decision.');
   const pair=known.get(row.pair_id);
   if(pair&&(row.left!==pair.left||row.right!==pair.right||row.origin!=='screen'))throw Error('Pair mismatch.');
   if(!pair){if(row.origin!=='manual')throw Error('Unknown pair origin.');extras.push(row);}
   next.set(row.pair_id,row);
  }
  prior=next;manual=extras;
  cards.forEach((card,i)=>{const row=prior.get(data.pairs[i].pair_id);for(const key of ['status','reason','evidence'])card.querySelector('.'+key).value=row?row[key]:'';});
  document.getElementById('reviewer').value=value.decisions[0]?.reviewer||'';
  message.textContent=`Loaded ${value.decisions.length} decisions. The CLI validates the exported file before grouping.`;
 }catch(error){message.textContent=error.message;}
};
</script></html>'''
        temporary.write_text(page, encoding="utf-8")
    return {"review_pairs": len(pairs), "original_images_linked": len(needed),
            "screen_sha256": screen["screen_sha256"], "decisions_recorded": 0}
