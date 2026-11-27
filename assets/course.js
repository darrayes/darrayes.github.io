/* Hover (or focus) previews for course pages: lecture cards list what is inside, topic chips show the
   section that teaches them, course-map stops summarize a unit. Same look as the §-link previews in the notes. */
(function(){
  let P={}; try{P=JSON.parse(document.getElementById("peeks").textContent);}catch(e){return;}
  const card=document.createElement("div"); card.id="xprev"; card.setAttribute("role","tooltip"); document.body.appendChild(card);
  let timer=null, cur=null;
  const hover=matchMedia("(hover:hover)").matches;
  function show(el){
    const h=P[el.dataset.peek]; if(!h) return;
    cur=el; card.innerHTML=h; card.classList.add("on");
    const hue=getComputedStyle(el).getPropertyValue("--hue"); card.style.setProperty("--hue",hue||"");
    const r=el.getBoundingClientRect(), big=el.classList.contains("lcard");
    const w=Math.min(big?340:360,innerWidth-24); card.style.width=w+"px";
    const hgt=card.offsetHeight;
    let left, top;
    const side=innerWidth-r.right-24;
    if(big && side>=260){ card.style.width=Math.min(w,side)+"px"; left=r.right+12; top=Math.max(8,Math.min(r.top,innerHeight-card.offsetHeight-8)); }
    else{
      left=Math.max(12,Math.min(big?r.left+24:r.left,innerWidth-w-12));
      const below=r.bottom+8; top=(below+hgt>innerHeight-8 && r.top-hgt-8>8)?r.top-hgt-8:below;
    }
    card.style.left=left+"px"; card.style.top=top+"px";
  }
  function hide(){clearTimeout(timer); card.classList.remove("on"); cur=null;}
  const pick=e=>e.target.closest&&e.target.closest("[data-peek]");
  document.addEventListener("mouseover",e=>{if(!hover) return; const el=pick(e); if(!el||el===cur) return;
    clearTimeout(timer); timer=setTimeout(()=>show(el),el.classList.contains("lcard")?320:160);});
  document.addEventListener("mouseout",e=>{const el=pick(e); if(!el) return;
    if(e.relatedTarget&&(el.contains(e.relatedTarget)||card.contains(e.relatedTarget))) return;
    clearTimeout(timer); timer=setTimeout(hide,120);});
  card.addEventListener("mouseenter",()=>clearTimeout(timer));
  card.addEventListener("mouseleave",()=>{timer=setTimeout(hide,120);});
  document.addEventListener("focusin",e=>{const el=pick(e); if(el) show(el);});
  document.addEventListener("focusout",e=>{if(pick(e)) hide();});
  // touch: first tap on a placeholder card or a non-link chip shows its preview
  document.addEventListener("click",e=>{const el=pick(e); if(!el){hide(); return;}
    if(el.tagName!=="A"){ e.preventDefault(); cur===el?hide():show(el); }});
  addEventListener("scroll",()=>{if(cur) hide();},{passive:true});
  addEventListener("keydown",e=>{if(e.key==="Escape") hide();});
})();
