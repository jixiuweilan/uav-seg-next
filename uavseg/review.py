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


def review_page(paths, document, screen, output, protected, *, practice=False):
    if practice and document['manifest'].get('purpose') != 'synthetic-review-practice':
        raise AuditError('practice pages require an explicitly synthetic manifest')
    pairs = validate_screen(screen, document)
    rows = training_rows(document)
    needed = sorted({row[side] for row in pairs.values() for side in ("left", "right")})
    if practice:
        needed = sorted(rows)
    previews, originals, thumbnails = {}, {}, {}
    # The artifact contains local file links; write it only to ignored local storage.
    if ".local" not in Path(output).resolve().parts:
        raise AuditError("review HTML with local original-image links must be under .local/")
    with new_output(output, protected) as temporary:
        for sample_id in needed:
            with verified_image(paths, rows[sample_id]) as image:
                thumb = image.resize((256, 256), Image.Resampling.LANCZOS)
                if practice:
                    stream = BytesIO()
                    image.save(stream, format="PNG")
                    originals[sample_id] = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
            thumbnails[sample_id] = thumb
            previews[sample_id] = jpeg_uri(thumb)
            if not practice:
                originals[sample_id] = (paths.folders["train_images"] / (sample_id + ".png")).resolve().as_uri()
        cards = []
        for index, row in enumerate(pairs.values()):
            left, right = row["left"], row["right"]
            aligned = Image.fromarray(transformed(np.asarray(thumbnails[right]), POSES.index(row["pose"])))
            images = "".join(
                f'<a class="original" href="{html.escape(originals[s])}" target="_blank" rel="noopener">'
                f'<img src="{previews[s]}" alt="样本 {html.escape(s)}"><span>{html.escape(s)} — 查看原图</span></a>'
                for s in (left, right))
            cross = "跨训练与验证划分" if row["cross_split"] else "同一划分或尚未划分"
            queue = "相似候选" if row["queue"] == "candidate" else "抽样对照"
            pose = {"r0": "不旋转", "r90": "逆时针旋转 90 度", "r180": "旋转 180 度",
                    "r270": "逆时针旋转 270 度", "flip-r0": "水平镜像",
                    "flip-r90": "水平镜像后逆时针旋转 90 度", "flip-r180": "水平镜像后旋转 180 度",
                    "flip-r270": "水平镜像后逆时针旋转 270 度"}[row["pose"]]
            cards.append(f'''<article data-index="{index}">
<h2>{index + 1}. {html.escape(left)} / {html.escape(right)}</h2>
<p>{queue} · {cross} · 纹理较少：{'是' if row['low_texture'] else '否'}</p>
<div class="images">{images}</div>
<details><summary>辅助对齐预览：右图{pose}</summary><img src="{jpeg_uri(aligned)}" alt="右图对齐后的辅助预览"><p>相似度仅供参考：归一化图像差异 {row['rms']}；哈希差异位数 {row['hamming']}。不能据此直接确认同一场景。</p></details>
<label>判断结果 <select class="status"><option value="">尚未查看</option><option value="confirmed">确认：同一场景或重复图像</option><option value="rejected">排除：不同场景</option><option value="uncertain">存疑：证据不足</option></select></label>
<label>判断依据 <select class="evidence"><option value="">请选择依据</option><option value="full_resolution_images">原始分辨率图像</option><option value="source_metadata">来源或航拍记录</option></select></label>
<label>理由或资料出处 <textarea class="reason" rows="2" placeholder="写出对应的道路、建筑等具体位置，或资料名称；证据不足时写明还缺什么。"></textarea></label>
</article>''')
        data = {"base": empty_decisions(document, screen), "pairs": list(pairs.values()),
                "training_ids": sorted(rows), "practice": practice}
        if practice:
            data["base"]["kind"] = "scene-practice-decisions"
        encoded = canonical(data).decode().replace("<", "\\u003c")
        page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练图像场景复核</title>
<style>body{font:16px system-ui;max-width:1000px;margin:24px auto;padding:0 16px;color:#172033;background:#f4f6f8}header,section,article{background:white;padding:20px;border:1px solid #d8dfe8;border-radius:8px;margin:16px 0}header{position:sticky;top:0;z-index:1}h1{font-size:24px}h2{font-size:18px}.images{display:flex;gap:16px;flex-wrap:wrap}img{width:256px;height:256px}a span{display:block}label{display:block;margin-top:12px}textarea{width:95%}button,input,select{font:inherit;padding:5px}#message{font-weight:bold}#manual-list{white-space:pre-wrap}</style>
<header><h1>训练图像场景复核</h1>
''' + ('<p><strong>合成图片练习：仅用于检查操作与说明，不是官方数据。导出记录不能用于真实场景分组。</strong></p>' if practice else '') + '''
<p>先查看两张原图，再记录判断。相似外观只是线索；不能确认时请选择“存疑”。页面不会上传文件或修改数据。</p>
<label>复核人 <input id="reviewer" placeholder="填写姓名或复核编号"></label>
<button id="export">下载复核记录</button> <label>恢复已下载的记录 <input type="file" id="import" accept=".json"></label>
<p id="message" role="status">记录尚未保存，请点击“下载复核记录”留存。</p></header>
<section><h2>如何判断</h2><p>确认：能指出多处对应的固定地物及其相对位置，或有可核验的同一地点记录。只有颜色、纹理或建筑样式相似不够。</p>
<p>排除：有明确的不同地点证据；没有找到相似点并不足以排除。存疑：证据不足或相互矛盾。尚未查看：还没有开始检查，不会作为已完成记录导出。</p>
<p>图像朝向、光照和裁剪可能变化；同一次航拍也可能经过不同地点。请说明具体证据。完成录入后下载文件，重新打开页面时用“恢复”继续。</p></section>
<section><h2>补充列表之外的关系</h2>
<p>已有额外的图像或来源证据时，选择两个样本编号并填写判断；列表中已有的关系请在对应卡片中填写。</p>
<label>第一个样本编号 <input id="manual-left" list="training-ids"></label>
<label>第二个样本编号 <input id="manual-right" list="training-ids"></label>
<datalist id="training-ids">''' + "".join(
            f'<option value="{html.escape(sample_id)}"></option>' for sample_id in sorted(rows)) + '''</datalist>
<label>判断结果 <select id="manual-status"><option value="">请选择结果</option><option value="confirmed">确认：同一场景或重复图像</option><option value="rejected">排除：不同场景</option><option value="uncertain">存疑：证据不足</option></select></label>
<label>判断依据 <select id="manual-evidence"><option value="">请选择依据</option><option value="full_resolution_images">原始分辨率图像</option><option value="source_metadata">来源或航拍记录</option></select></label>
<label>理由或资料出处 <textarea id="manual-reason" rows="2"></textarea></label>
<button id="manual-add">添加补充关系</button> <button id="manual-remove">撤销最后一条补充关系</button>
<p id="manual-list">尚无补充关系。</p></section>
<section id="original-view" hidden><h2>原始分辨率图像</h2><p>练习图内嵌在本文件中，可滚动查看完整细节。</p><div style="overflow:auto"><img id="original-image" style="width:auto;height:auto;max-width:none" alt="练习原图"></div></section>
''' + ('<section><h2>全部练习原图</h2><p>仅为几何图形模拟，不提供真实地点或航拍记录。补充关系前可在这里查看原图。</p>' + ''.join(
            f'<p><a class="original" href="{html.escape(originals[s])}">{html.escape(s)} — 查看原图</a></p>'
            for s in needed) + '</section>' if practice else '') + "\n".join(cards) + '<script type="application/json" id="data">' + encoded + '''</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const cards=[...document.querySelectorAll('article')];
const message=document.getElementById('message');
const knownPairs=new Map(data.pairs.map(pair=>[pair.pair_id,pair]));
const knownIds=new Set(data.training_ids);
let prior=new Map(), manual=[];
function showError(error){
 message.textContent=/[一-鿿]/.test(error.message)?error.message:'操作未完成，请记录浏览器名称和操作步骤，在飞书任务中反馈。';
}
const nonempty=value=>typeof value==='string'&&value.trim();
const labels={confirmed:'确认',rejected:'排除',uncertain:'存疑'};
function timezoneDate(value){
 if(typeof value!=='string')return false;
 const match=value.match(/^(\\d{4})-(\\d{2})-(\\d{2})T(\\d{2}):(\\d{2}):(\\d{2})(?:\\.\\d{1,6})?(Z|[+-]\\d{2}:\\d{2})$/);
 if(!match)return false;
 const [,year,month,day,hour,minute,second,zone]=match;
 const days=new Date(Date.UTC(Number(year),Number(month),0)).getUTCDate();
 return Number(year)>=1&&Number(month)>=1&&Number(month)<=12&&Number(day)>=1&&Number(day)<=days
  &&Number(hour)<24&&Number(minute)<60&&Number(second)<60
  &&(zone==='Z'||(Number(zone.slice(1,3))<24&&Number(zone.slice(4))<60))&&Number.isFinite(Date.parse(value));
}
if(data.practice){
 document.querySelectorAll('a.original').forEach(link=>{link.onclick=event=>{
  event.preventDefault();
  document.getElementById('original-image').src=link.href;
  const view=document.getElementById('original-view');view.hidden=false;view.scrollIntoView();
 };});
}
async function pairId(left,right){
 if(typeof crypto==='undefined'||!crypto.subtle)throw Error('浏览器无法校验记录，请下载本文件后用电脑上的浏览器打开；仍失败时反馈浏览器名称。');
 const bytes=new TextEncoder().encode(JSON.stringify([left,right])+'\\n');
 const digest=await crypto.subtle.digest('SHA-256',bytes);
 return [...new Uint8Array(digest)].map(value=>value.toString(16).padStart(2,'0')).join('');
}
function renderManual(){
 document.getElementById('manual-list').textContent=manual.length
 ?manual.map((row,index)=>`${index+1}. ${row.left} / ${row.right} — ${labels[row.status]}；${row.reason}（复核人：${row.reviewer}）`).join('\\n')
  :'尚无补充关系。';
}
document.getElementById('manual-add').onclick=async()=>{
 try{
  const reviewer=document.getElementById('reviewer').value.trim();
  const ids=[document.getElementById('manual-left').value.trim(),document.getElementById('manual-right').value.trim()].sort();
  const status=document.getElementById('manual-status').value,evidence=document.getElementById('manual-evidence').value;
  const reason=document.getElementById('manual-reason').value.trim();
  if(!reviewer||!reason||!['confirmed','rejected','uncertain'].includes(status)||!['full_resolution_images','source_metadata'].includes(evidence))throw Error('请填写复核人、判断结果、依据和理由。');
  if(ids[0]===ids[1]||!knownIds.has(ids[0])||!knownIds.has(ids[1]))throw Error('请选择两个不同且在本批次清单中的样本编号。');
  const key=await pairId(ids[0],ids[1]);
  if(knownPairs.has(key))throw Error('此关系已在列表中，请到对应卡片填写。');
  if(manual.some(row=>row.pair_id===key))throw Error('此补充关系已经记录，请勿重复添加。');
  manual.push({pair_id:key,left:ids[0],right:ids[1],status,reason,evidence,origin:'manual',reviewer,reviewed_at:new Date().toISOString()});
  renderManual();message.textContent='已添加补充关系，请下载复核记录保存。';
 }catch(error){showError(error);}
};
document.getElementById('manual-remove').onclick=()=>{
 if(!manual.length){message.textContent='尚无可撤销的补充关系。';return;}
 manual.pop();renderManual();message.textContent='已撤销最后一条补充关系，请下载记录保存更改。';
};
document.getElementById('export').onclick=()=>{
 try {
  const reviewer=document.getElementById('reviewer').value.trim();
  const decisions=[...manual];
  cards.forEach((card,i)=>{
   const status=card.querySelector('.status').value;
   if(!status)return;
   const reason=card.querySelector('.reason').value.trim(),evidence=card.querySelector('.evidence').value;
   if(!reviewer||!reason||!evidence)throw Error('每条判断都需要填写复核人、依据和理由。');
   const pair=data.pairs[i],old=prior.get(pair.pair_id);
   const unchanged=old&&old.status===status&&old.reason===reason&&old.evidence===evidence;
   decisions.push({pair_id:pair.pair_id,left:pair.left,right:pair.right,status,reason,evidence,origin:'screen',reviewer:unchanged?old.reviewer:reviewer,reviewed_at:unchanged?old.reviewed_at:new Date().toISOString()});
  });
  const blob=new Blob([JSON.stringify({...data.base,decisions},null,2)+'\\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=data.practice?'练习复核记录.json':'场景复核记录.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  message.textContent=`已下载 ${decisions.length} 条记录。请为每次修改保留独立文件。`;
 }catch(error){showError(error);}
};
document.getElementById('import').onchange=async event=>{
 try{
  if(!event.target.files.length)return;
  let value;
  try{value=JSON.parse(await event.target.files[0].text());}catch{throw Error('无法读取记录，请选择本页面下载的 JSON 文件。');}
  if(!value||typeof value!=='object')throw Error('记录格式不正确。');
  for(const key of ['schema_version','kind','manifest_sha256','screen_sha256'])if(value[key]!==data.base[key])throw Error('记录版本或批次不匹配，不能在本页面恢复。');
  if(!Array.isArray(value.decisions))throw Error('记录列表格式不正确。');
  const next=new Map(),extras=[];
  for(const row of value.decisions){
   if(!row||typeof row!=='object'||!nonempty(row.pair_id)||next.has(row.pair_id)||!['confirmed','rejected','uncertain'].includes(row.status)||!nonempty(row.reason)||!nonempty(row.reviewer)||!['full_resolution_images','source_metadata'].includes(row.evidence))throw Error('记录重复或必填内容不完整。');
   if(!timezoneDate(row.reviewed_at))throw Error('复核时间必须是含时区的有效时间，例如 2026-09-13T17:30:00+08:00。');
   if(!knownIds.has(row.left)||!knownIds.has(row.right)||row.left>=row.right)throw Error('记录必须引用清单中两个不同且已排序的样本编号。');
   if(row.pair_id!==await pairId(row.left,row.right))throw Error('关系编号与样本编号不匹配。');
   const pair=knownPairs.get(row.pair_id);
   if(pair&&(row.left!==pair.left||row.right!==pair.right||row.origin!=='screen'))throw Error('列表关系或来源不匹配。');
   if(!pair){if(row.origin!=='manual')throw Error('列表外的关系必须注明为人工补充。');extras.push(row);}
   next.set(row.pair_id,row);
  }
  prior=next;manual=extras;
  renderManual();
  cards.forEach((card,i)=>{const row=prior.get(data.pairs[i].pair_id);for(const key of ['status','reason','evidence'])card.querySelector('.'+key).value=row?row[key]:'';});
  document.getElementById('reviewer').value=value.decisions[0]?.reviewer||'';
  message.textContent=`已恢复 ${value.decisions.length} 条记录。提交后仍需工具校验，复核完成不代表可以开始训练。`;
 }catch(error){showError(error);}
};
</script></html>'''
        temporary.write_text(page, encoding="utf-8")
    return {"review_pairs": len(pairs), "original_images_linked": len(needed),
            "screen_sha256": screen["screen_sha256"], "decisions_recorded": 0}
