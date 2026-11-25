/* =====================================================================
   lib.js — shared helpers for the Advanced DL notes (plain JS, no libraries).
   Colour roles (same as the Transformers notes): Q blue, K orange, V aqua,
   encoder steel blue, decoder violet, FFN amber, masked = hatched grey.
   Demos register a function in FIG under a name; any element with
   data-fig="name" is handed to that function on load.
   ===================================================================== */
const NS = "http://www.w3.org/2000/svg";
const C = {q:"#2a78d6",k:"#eb6834",v:"#1baf7a",enc:"#3d6fb6",dec:"#6250d6",ffn:"#b77d00",norm:"#7a7870",
           ink:"#1b1b1a",ink2:"#52514e",mut:"#85837d",line:"#e4e2dc",soft:"#f4f3ef",paper:"#ffffff"};
const FIG = {};
const fmt = (x,d=4)=> (typeof x==="string")?x : (!isFinite(x)? (x>0?"∞":"−∞") : (Math.abs(x)<5e-5?"0":(+x.toFixed(d)).toString().replace("-","−")));
const softmax = a => {const f=a.filter(isFinite); const m=Math.max(...f); const e=a.map(x=>isFinite(x)?Math.exp(x-m):0); const s=e.reduce((p,c)=>p+c,0); return e.map(x=>x/s);};
const sigmoid = z => 1/(1+Math.exp(-z));
function svg(w,h,inner,cls=""){return `<svg viewBox="0 0 ${w} ${h}" width="${w}" class="${cls}" role="img">${inner}</svg>`;}
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

/* ---------- colour scales ---------- */
function mix(a,b,t){const p=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));const A=p(a),B=p(b);return "#"+A.map((x,i)=>Math.round(x+(B[i]-x)*t).toString(16).padStart(2,"0")).join("");}
function lum(h){const [r,g,b]=[1,3,5].map(i=>parseInt(h.slice(i,i+2),16)/255);return .2126*r+.7152*g+.0722*b;}
const DIV_POS="#2f6db5", DIV_NEG="#c8553d", MID="#f3f2ee", SEQ_LO="#f6f8fc", SEQ_HI="#1f4f8f";
function colDiv(v,m){const t=Math.min(1,Math.abs(v)/(m||1)); return mix(MID, v>=0?DIV_POS:DIV_NEG, t*.85);}
function colSeq(v,m){return mix(SEQ_LO,SEQ_HI,Math.min(1,Math.max(0,v)/(m||1)));}

/* ---------- tooltip ---------- */
document.addEventListener("mousemove",e=>{const tip=document.getElementById("tip"); if(!tip) return;
  const t=e.target.closest&&e.target.closest("[data-tip]");
  if(t){tip.innerHTML=t.getAttribute("data-tip");tip.style.opacity=1;tip.style.left=Math.min(e.clientX+14,innerWidth-270)+"px";tip.style.top=(e.clientY+14)+"px";}
  else tip.style.opacity=0;});

/* ---------- matrices: <div class="mat" data-m='[[..]]' data-r='[..]' data-c='[..]' data-t="title" data-kind="div|seq|plain"> ---------- */
function renderMat(el){
  const M=JSON.parse(el.dataset.m).map(r=>r.map(x=>x==="-inf"?-Infinity:x));
  const R=el.dataset.r?JSON.parse(el.dataset.r).map(esc):null, Cc=el.dataset.c?JSON.parse(el.dataset.c).map(esc):null;
  const kind=el.dataset.kind||"div", d=+(el.dataset.d||4);
  const fin=M.flat().filter(isFinite), mx=+(el.dataset.max||Math.max(...fin.map(Math.abs)))||1;
  const hl=new Set((el.dataset.hl||"").split(";").filter(Boolean));
  let h=""; if(el.dataset.t) h+=`<div class="mt">${el.dataset.t}</div>`;
  h+="<table>"; if(Cc) h+="<tr>"+(R?"<th></th>":"")+Cc.map(c=>`<th>${c}</th>`).join("")+"</tr>";
  M.forEach((row,i)=>{h+="<tr>"+(R?`<th class="r">${R[i]}</th>`:"");
    row.forEach((v,j)=>{
      const lab=(R?R[i]:"row "+(i+1))+" · "+(Cc?Cc[j]:"col "+(j+1));
      if(!isFinite(v)){h+=`<td class="inf${hl.has(i+","+j)?" hl":""}" data-tip="${lab}: masked (−∞)">−∞</td>`;return;}
      const bg= kind==="plain"?"#f6f5f2": kind==="seq"?colSeq(v,mx):colDiv(v,mx);
      const fg= lum(bg)<.5?"#fff":C.ink;
      h+=`<td class="${hl.has(i+","+j)?"hl":""}" style="background:${bg};color:${fg}" data-tip="${lab}: ${fmt(v,d)}">${fmt(v,d)}</td>`;});
    h+="</tr>";});
  h+="</table>";
  if(el.dataset.shape!=="no") h+=`<div class="shp">${M.length} × ${M[0].length}</div>`;
  el.innerHTML=h;
}

/* ---------- generic line plot ---------- */
function linePlot(o){
  const W=o.w||560,H=o.h||260,L=o.left||48,Rm=o.right||16,T=o.top||14,B=o.bottom||40;
  const [x0,x1]=o.x,[y0,y1]=o.y; const X=x=>L+(x-x0)/(x1-x0)*(W-L-Rm), Y=y=>H-B-(y-y0)/(y1-y0)*(H-T-B);
  let s="";
  (o.yticks||[]).forEach(t=>{s+=`<line class="grid" x1="${L}" x2="${W-Rm}" y1="${Y(t)}" y2="${Y(t)}"/><text class="xs" x="${L-6}" y="${Y(t)+3}" text-anchor="end">${o.yfmt?o.yfmt(t):fmt(t,3)}</text>`;});
  (o.xticks||[]).forEach(t=>{s+=`<line class="grid" x1="${X(t)}" x2="${X(t)}" y1="${T}" y2="${H-B}" opacity=".5"/><text class="xs" x="${X(t)}" y="${H-B+15}" text-anchor="middle">${o.xfmt?o.xfmt(t):fmt(t,3)}</text>`;});
  if(y0<0&&y1>0) s+=`<line class="axis" x1="${L}" x2="${W-Rm}" y1="${Y(0)}" y2="${Y(0)}"/>`;
  s+=`<line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H-B}"/><line class="axis" x1="${L}" x2="${W-Rm}" y1="${H-B}" y2="${H-B}"/>`;
  if(o.xlabel) s+=`<text class="sm" x="${(L+W-Rm)/2}" y="${H-6}" text-anchor="middle">${o.xlabel}</text>`;
  if(o.ylabel) s+=`<text class="sm" transform="translate(12,${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${o.ylabel}</text>`;
  if(o.under) s+=o.under(X,Y);
  (o.series||[]).forEach(se=>{
    let pts=se.pts; if(se.f){pts=[];const n=se.n||300;for(let i=0;i<=n;i++){const x=x0+(x1-x0)*i/n;const y=se.f(x);if(isFinite(y))pts.push([x,y]);}}
    s+=`<path d="${pts.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(Math.max(y0,Math.min(y1,p[1]))).toFixed(1)).join("")}" fill="none" stroke="${se.color}" stroke-width="${se.wd||2}" ${se.dash?'stroke-dasharray="5 4"':""}/>`;
    if(se.label&&se.lx!==undefined) s+=`<text class="sm b" x="${X(se.lx)}" y="${Y(se.ly)}" style="fill:${se.color}">${se.label}</text>`;
  });
  (o.marks||[]).forEach(m=>{s+=`<circle cx="${X(m.x)}" cy="${Y(m.y)}" r="5" fill="${m.color||C.ink}" stroke="#fff" stroke-width="2" data-tip="${m.tip||""}"/>`;});
  return svg(W,H,s+(o.extra?o.extra(X,Y):""));
}

/* ---------- page chrome: table of contents, scroll-spy, progress bar, filter ---------- */
function tocChrome(){
  const list=document.getElementById("toclist"); if(!list) return;
  const links=[...list.querySelectorAll("a")],groups=[...list.querySelectorAll(".pgroup")];
  const KEY="toc-collapsed:"+location.pathname;let saved=[];
  try{saved=JSON.parse(localStorage.getItem(KEY)||"[]");}catch(e){}
  groups.forEach((g,i)=>{if(saved.includes(i))g.classList.add("closed");
    g.querySelector(".ptitle").onclick=()=>{g.classList.toggle("closed");
      try{localStorage.setItem(KEY,JSON.stringify(groups.map((x,j)=>x.classList.contains("closed")?j:-1).filter(j=>j>=0)));}catch(e){}};});
  document.getElementById("tocq").oninput=e=>{const q=e.target.value.toLowerCase();
    links.forEach(a=>a.style.display=a.textContent.toLowerCase().includes(q)?"":"none");
    list.classList.toggle("filtering",!!q);
    groups.forEach(g=>g.style.display=!q||[...g.querySelectorAll("a")].some(a=>a.style.display!=="none")?"":"none");};
  links.forEach(a=>a.addEventListener("click",()=>document.body.classList.remove("menu-open")));
  const bar=document.querySelector("#progress div");
  const secs=links.map(a=>document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
  const spy=()=>{const y=scrollY+120;let cur=secs[0];secs.forEach(s=>{if(s.offsetTop<=y)cur=s;});
    links.forEach(a=>a.classList.toggle("on",a.getAttribute("href")==="#"+cur.id));
    groups.forEach(g=>g.classList.toggle("has-on",!!g.querySelector("a.on")));
    const h=document.documentElement.scrollHeight-innerHeight;bar.style.width=(h>0?100*scrollY/h:0)+"%";};
  addEventListener("scroll",spy,{passive:true});spy();
  let lastOn=null;setInterval(()=>{const a=list.querySelector("a.on");if(a&&a!==lastOn){lastOn=a;const r=a.getBoundingClientRect(),tr=document.getElementById("toc").getBoundingClientRect();if(r.top<tr.top+40||r.bottom>tr.bottom-20)a.scrollIntoView({block:"center"});}},400);
}

function init(){
  document.querySelectorAll("[data-fig]").forEach(el=>{try{FIG[el.dataset.fig](el);}catch(e){console.error(el.dataset.fig,e);el.textContent="(figure failed to load: "+el.dataset.fig+")";}});
  document.querySelectorAll(".mat[data-m]").forEach(renderMat);
  tocChrome();
}
addEventListener("DOMContentLoaded",init);
