/* Sliding-window mask and multi-layer reach (§ Sliding-window and sparse attention). */
FIG.windowMask = el => {
  const T = 12;
  el.innerHTML = `<div class="ctrl"><label>window w <input type="range" min="1" max="12" step="1" value="3"><output></output></label>
    <span class="lay"><button data-l="1">1 layer</button><button data-l="2" class="on">2 layers</button><button data-l="3">3 layers</button></span></div>
    <div class="panel"><div class="mask"></div><div class="reach"></div></div>`;
  const r = el.querySelector("input"), o = el.querySelector("output");
  let L = 2, sel = T - 1;
  function draw() {
    const w = +r.value; o.textContent = w;
    const cs = 20, x0 = 30, y0 = 24; let g = `<text class="xs" x="${x0 + T * cs / 2}" y="14" text-anchor="middle">Key position j →</text>`;
    let pairs = 0;
    for (let i = 0; i < T; i++) { g += `<text class="xs" x="${x0 - 6}" y="${y0 + i * cs + 14}" text-anchor="end">${i}</text>`;
      for (let j = 0; j < T; j++) { const on = j <= i && j > i - w; if (on) pairs++;
        g += `<rect x="${x0 + j * cs}" y="${y0 + i * cs}" width="${cs - 2}" height="${cs - 2}" rx="2" class="${on ? "fq" : "fmask"}" ${i === sel ? 'stroke-width="2.2"' : ""}/>`; } }
    g += `<text class="xs" x="${x0}" y="${y0 + T * cs + 16}">computed pairs: ${pairs} (full causal: ${T * (T + 1) / 2})</text>`;
    el.querySelector(".mask").innerHTML = svg(x0 + T * cs + 10, y0 + T * cs + 24, g);
    // reach: which tokens can influence token sel after L layers (each layer reaches back w-1)
    const reach = new Array(T).fill(0); // layer index at which first reached
    reach[sel] = 0.5;
    for (let l = 1; l <= L; l++) { const back = l * (w - 1); for (let j = Math.max(0, sel - back); j < sel; j++) if (!reach[j]) reach[j] = l; }
    const bw = 34; let h = `<text class="sm b" x="10" y="16">who can reach token ${sel} (click a token)</text>`;
    const cols = ["", C.q, C.dec, C.k];
    for (let j = 0; j < T; j++) { const lv = reach[j];
      h += `<g class="tok" data-j="${j}" style="cursor:pointer"><rect x="${10 + j * bw}" y="30" width="${bw - 4}" height="30" rx="5" fill="${j === sel ? C.ink : lv ? cols[lv] : "#f1f0ec"}" fill-opacity="${j === sel ? 1 : lv ? .85 : 1}" stroke="${C.line}"/>
            <text x="${10 + j * bw + (bw - 4) / 2}" y="50" text-anchor="middle" class="sm" style="fill:${j === sel || lv ? "#fff" : C.ink2}">${j}</text></g>`; }
    for (let l = 1; l <= L; l++) h += `<rect x="${10 + (l - 1) * 130}" y="76" width="12" height="12" rx="3" fill="${cols[l]}"/><text class="xs" x="${26 + (l - 1) * 130}" y="86">reached by layer ${l}</text>`;
    h += `<text class="xs" x="10" y="112">reach after ${L} layer${L > 1 ? "s" : ""}: up to L·(w − 1) = ${L}·${w - 1} = ${L * (w - 1)} positions back</text>`;
    el.querySelector(".reach").innerHTML = svg(10 + T * bw + 10, 122, h);
    el.querySelectorAll(".tok").forEach(t => t.onclick = () => { sel = +t.dataset.j; draw(); });
  }
  r.oninput = draw;
  el.querySelectorAll(".lay button").forEach(b => b.onclick = () => { L = +b.dataset.l; el.querySelectorAll(".lay button").forEach(x => x.classList.toggle("on", x === b)); draw(); });
  draw();
};
