/* Online softmax stepper and FlashAttention tiling (§ Online softmax, § FlashAttention).
   The scores and values are the worked example of the notes (also computed in compute.py). */
FIG.onlineSoftmax = el => {
  const s = [1, 3, 2, 0, 4, 1], V = [[1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [1, -1]], bs = 2, nb = s.length / bs;
  el.innerHTML = `<div class="ctrl"><button class="prev">◂ Back</button><button class="next">Next block ▸</button><button class="reset">Reset</button></div><div class="view"></div>`;
  let k = 0;
  function state(upto) { let m = -Infinity, l = 0, o = [0, 0], log = [];
    for (let b = 0; b < upto; b++) { const xs = s.slice(b * bs, b * bs + bs), vs = V.slice(b * bs, b * bs + bs);
      const mn = Math.max(m, ...xs), corr = isFinite(m) ? Math.exp(m - mn) : 0, e = xs.map(x => Math.exp(x - mn));
      const ln = corr * l + e.reduce((a, c) => a + c, 0); const on = o.map((oi, d) => (corr * l * oi + e.reduce((a, ej, j) => a + ej * vs[j][d], 0)) / ln);
      log.push({ mold: m, m: mn, corr, e, l: ln, o: on }); m = mn; l = ln; o = on; }
    return { m, l, o, log }; }
  function draw() {
    const st = state(k), w = softmax(s), direct = [0, 1].map(d => w.reduce((a, wj, j) => a + wj * V[j][d], 0));
    let g = ""; const cw = 74;
    s.forEach((x, j) => { const b = Math.floor(j / bs), seen = b < k, cur = b === k - 1;
      g += `<rect x="${20 + j * cw}" y="30" width="${cw - 8}" height="34" rx="6" fill="${seen ? C.k : "#f1f0ec"}" fill-opacity="${seen ? .18 : 1}" stroke="${cur ? C.ink : C.line}" stroke-width="${cur ? 2.5 : 1}"/>
            <text x="${20 + j * cw + (cw - 8) / 2}" y="52" text-anchor="middle" class="b">${x}</text>
            <text class="xs" x="${20 + j * cw + (cw - 8) / 2}" y="80" text-anchor="middle">v = [${V[j].join(", ").replace(/-/g, "−")}]</text>`; });
    for (let b = 0; b < nb; b++) g += `<text class="xs" x="${20 + b * bs * cw + cw - 4}" y="20" text-anchor="middle">block ${b + 1}</text>`;
    const L = st.log[k - 1];
    const box = (x, t, v, col) => `<rect x="${x}" y="100" width="140" height="56" rx="8" class="box"/><text class="xs" x="${x + 70}" y="118" text-anchor="middle">${t}</text><text class="b" x="${x + 70}" y="144" text-anchor="middle" style="fill:${col || C.ink}">${v}</text>`;
    g += box(20, "running max m", k ? fmt(st.m, 4) : "−∞") + box(180, "running sum ℓ", fmt(st.l, 4)) + box(340, "output o", `[${st.o.map(v => fmt(v, 4)).join(", ")}]`) +
         box(500, "correction e^(m_old − m_new)", L ? fmt(L.corr, 4) : "—", L && L.corr < 1 && L.corr > 0 ? C.k : C.ink);
    g += `<text class="sm" x="20" y="186">${k === 0 ? "Nothing read yet: m = −∞, ℓ = 0, o = [0, 0]." : k < nb ? `After block ${k}: ${L.corr === 0 ? "first block, nothing to correct" : L.corr === 1 ? "maximum unchanged, correction = 1" : "a new maximum arrived, so the old ℓ and o are scaled down"}.` : "All blocks read."}</text>`;
    g += `<text class="sm" x="20" y="210">direct softmax(s)·V over the whole row = [${direct.map(v => fmt(v, 4)).join(", ")}] ${k === nb ? (Math.abs(direct[0] - st.o[0]) < 1e-12 && Math.abs(direct[1] - st.o[1]) < 1e-12 ? "✓ identical" : "✗") : ""}</text>`;
    el.querySelector(".view").innerHTML = svg(660, 222, g);
  }
  el.querySelector(".next").onclick = () => { k = Math.min(nb, k + 1); draw(); };
  el.querySelector(".prev").onclick = () => { k = Math.max(0, k - 1); draw(); };
  el.querySelector(".reset").onclick = () => { k = 0; draw(); };
  draw();
};

FIG.flashTiles = el => {
  const T = 16, Br = 4, Bc = 4, nb = T / Br, nc = T / Bc;
  el.innerHTML = `<div class="ctrl"><button class="next">Next tile ▸</button><button class="all">Run all</button><button class="reset">Reset</button>
    <label><input type="checkbox" class="causal" checked> causal (skip masked tiles)</label></div><div class="panel"><div class="grid"></div><div class="readout"></div></div>`;
  let step = 0;
  const tiles = () => { const out = []; const causal = el.querySelector(".causal").checked;
    for (let b = 0; b < nb; b++) for (let c = 0; c < nc; c++) if (!causal || c <= b) out.push([b, c]); return out; };
  function draw() {
    const tl = tiles(), cur = tl[step - 1], causal = el.querySelector(".causal").checked, cs = 18, x0 = 40, y0 = 30;
    let g = `<text class="xs" x="${x0 + T * cs / 2}" y="18" text-anchor="middle">Keys / Values (columns) →</text><text class="xs" transform="translate(22,${y0 + T * cs / 2}) rotate(-90)" text-anchor="middle">Queries (rows) →</text>`;
    const done = new Set(tl.slice(0, step).map(t => t.join(",")));
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const b = Math.floor(i / Br), c = Math.floor(j / Bc), key = b + "," + c, masked = causal && j > i;
      let fill = "#f6f5f2"; if (done.has(key)) fill = masked ? "#ecebe6" : "#d9e6f7"; if (cur && cur[0] === b && cur[1] === c) fill = masked ? "#e4e2dc" : C.q;
      g += `<rect x="${x0 + j * cs}" y="${y0 + i * cs}" width="${cs - 1}" height="${cs - 1}" fill="${masked && !(cur && cur[0] === b && cur[1] === c) && !done.has(key) ? "url(#hatch)" : fill}"/>`; }
    for (let b = 0; b <= nb; b++) g += `<line x1="${x0}" x2="${x0 + T * cs}" y1="${y0 + b * Br * cs}" y2="${y0 + b * Br * cs}" stroke="${C.ink2}"/><line y1="${y0}" y2="${y0 + T * cs}" x1="${x0 + b * Bc * cs}" x2="${x0 + b * Bc * cs}" stroke="${C.ink2}"/>`;
    if (cur) { g += `<rect x="${x0 - 8}" y="${y0 + cur[0] * Br * cs}" width="5" height="${Br * cs}" fill="${C.q}"/><rect x="${x0 + cur[1] * Bc * cs}" y="${y0 - 8}" width="${Bc * cs}" height="5" fill="${C.k}"/>`; }
    el.querySelector(".grid").innerHTML = svg(x0 + T * cs + 10, y0 + T * cs + 10, g);
    const total = tl.length, full = nb * nc;
    el.querySelector(".readout").innerHTML = `T = ${T}, tiles of ${Br} × ${Bc}<br>tiles computed: <b>${step}</b> of ${total}${causal ? ` (${full - total} masked tiles skipped)` : ""}<br>` +
      (cur ? `now in SRAM: Q block ${cur[0] + 1}, K/V block ${cur[1] + 1}, one ${Br} × ${Bc} score tile<br>then: update m, ℓ, o of rows ${cur[0] * Br + 1}–${cur[0] * Br + Br}` : "nothing loaded yet") +
      `<br><br>scores written to slow memory: <b>0</b><br>standard attention would write ${T * T} (and read them back twice)`;
  }
  el.querySelector(".next").onclick = () => { step = Math.min(tiles().length, step + 1); draw(); };
  el.querySelector(".all").onclick = () => { step = tiles().length; draw(); };
  el.querySelector(".reset").onclick = () => { step = 0; draw(); };
  el.querySelector(".causal").onchange = () => { step = 0; draw(); };
  draw();
};
