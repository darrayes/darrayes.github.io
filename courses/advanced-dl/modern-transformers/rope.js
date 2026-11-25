/* RoPE demos: clock dials (§ RoPE across all frequencies) and context extension (§ Extending the context window). */
FIG.ropeDials = el => {
  el.innerHTML = `<div class="ctrl"><label>position p <input type="range" min="0" max="2000" step="1" value="5"><output></output></label>
    <label>base <select><option value="10000">10,000</option><option value="500000">500,000</option></select></label>
    <button data-v="1">p = 1</button><button data-v="5">p = 5</button><button data-v="100">p = 100</button><button data-v="1000">p = 1000</button></div><div class="dials"></div>`;
  const r = el.querySelector("input"), o = el.querySelector("output"), sel = el.querySelector("select"), box = el.querySelector(".dials");
  const dk = 8;
  function draw() {
    const p = +r.value, base = +sel.value; o.textContent = p;
    let s = ""; const R = 52, gap = 170;
    for (let k = 0; k < dk / 2; k++) {
      const th = Math.pow(base, -2 * k / dk), ang = p * th, turns = ang / (2 * Math.PI);
      const cx = 80 + k * gap, cy = 80, a = ang % (2 * Math.PI);
      const x = cx + R * Math.cos(-a + 0), y = cy + R * Math.sin(-a);
      s += `<circle cx="${cx}" cy="${cy}" r="${R}" class="box"/><line class="grid" x1="${cx}" y1="${cy}" x2="${cx + R}" y2="${cy}"/>`;
      for (let t = 0; t < 12; t++) { const b = t * Math.PI / 6; s += `<line x1="${cx + (R - 5) * Math.cos(b)}" y1="${cy + (R - 5) * Math.sin(b)}" x2="${cx + R * Math.cos(b)}" y2="${cy + R * Math.sin(b)}" stroke="${C.line}"/>`; }
      s += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="${C.q}" stroke-width="3.5" stroke-linecap="round"/><circle cx="${cx}" cy="${cy}" r="4" fill="${C.ink}"/>`;
      s += `<text class="sm b" x="${cx}" y="${cy + R + 20}" text-anchor="middle">pair r = ${k}</text>`;
      s += `<text class="xs" x="${cx}" y="${cy + R + 35}" text-anchor="middle">θ = ${th < 1e-3 ? th.toExponential(2) : fmt(th, 4)} rad/pos</text>`;
      s += `<text class="xs" x="${cx}" y="${cy + R + 49}" text-anchor="middle">angle ${fmt(ang, 3)} rad = ${fmt(turns, 3)} turns</text>`;
      s += `<text class="xs" x="${cx}" y="${cy + R + 63}" text-anchor="middle">wavelength ${fmt(2 * Math.PI / th, 1)}</text>`;
    }
    box.innerHTML = svg(680, 210, s);
  }
  r.oninput = draw; sel.onchange = draw;
  el.querySelectorAll("button").forEach(b => b.onclick = () => { r.value = b.dataset.v; draw(); });
  draw();
};

FIG.ctxExt = el => {
  el.innerHTML = `<div class="ctrl"><label>s = L_new / L_train <input type="range" min="1" max="16" step="1" value="4"><output></output></label>
    <span class="meth"><button data-m="orig" class="on">original</button><button data-m="pi">PI</button><button data-m="ntk">NTK</button><button data-m="yarn">YaRN</button></span></div>
    <div class="panel"><div class="plot"></div><div class="readout"></div></div>`;
  const r = el.querySelector("input"), o = el.querySelector("output"), plot = el.querySelector(".plot"), ro = el.querySelector(".readout");
  const dk = 128, base = 10000, Ltr = 4096; let meth = "orig";
  const th = [...Array(dk / 2).keys()].map(k => Math.pow(base, -2 * k / dk));
  function thetaNew(s) {
    if (meth === "orig") return th.slice();
    if (meth === "pi") return th.map(t => t / s);
    if (meth === "ntk") { const b2 = base * Math.pow(s, dk / (dk - 2)); return th.map((t, k) => Math.pow(b2, -2 * k / dk)); }
    // YaRN "NTK-by-parts": ramp on the number of turns made during training (alpha = 1, beta = 32)
    return th.map(t => { const turns = Ltr * t / (2 * Math.PI); const g = Math.min(1, Math.max(0, (turns - 1) / (32 - 1))); return (1 - g) * t / s + g * t; });
  }
  function draw() {
    const s = +r.value; o.textContent = s; const Lnew = s * Ltr, tn = thetaNew(s);
    const trained = th.map(t => Ltr * t / (2 * Math.PI)), now = tn.map(t => Lnew * t / (2 * Math.PI));
    const lg = v => Math.log10(Math.max(v, 1e-3));
    let unseen = 0; const firstSlow = trained.findIndex(v => v < 1);
    const pts = now.map((v, k) => { const bad = trained[k] < 1 && v > trained[k] * 1.0001; if (bad) unseen++; return { k, v, bad }; });
    plot.innerHTML = linePlot({ w: 560, h: 290, x: [0, 63], y: [-1, 4], xticks: [0, 16, 32, 48, 63], yticks: [-1, 0, 1, 2, 3, 4],
      yfmt: t => t === 0 ? "1" : "10" + ["⁻¹", "", "¹", "²", "³", "⁴"][t + 1], xlabel: "pair r (fast → slow)", ylabel: "turns over the sequence",
      under: (X, Y) => `<rect x="${X(firstSlow - 0.5)}" y="${Y(4)}" width="${X(63) - X(firstSlow - 0.5)}" height="${Y(-1) - Y(4)}" fill="#fdf0ea" opacity=".6"/>
        <line x1="${X(firstSlow - 0.5)}" x2="${X(firstSlow - 0.5)}" y1="${Y(4)}" y2="${Y(-1)}" stroke="${C.k}" stroke-dasharray="4 4"/>
        <text class="xs" x="${X(firstSlow - 0.5) + 6}" y="${Y(3.7)}">less than 1 turn in training →</text>`,
      series: [{ pts: trained.map((v, k) => [k, lg(v)]), color: "#a9a69d", wd: 3 }, { pts: now.map((v, k) => [k, lg(v)]), color: C.q, wd: 1.5 }],
      extra: (X, Y) => pts.map(p => `<circle cx="${X(p.k)}" cy="${Y(Math.max(-1, Math.min(4, lg(p.v))))}" r="3.2" fill="${p.bad ? C.k : C.q}" data-tip="pair ${p.k}: ${fmt(p.v, 3)} turns now, ${fmt(trained[p.k], 3)} in training"/>`).join("") +
        `<rect x="${X(2)}" y="${Y(0.25)}" width="12" height="4" fill="#a9a69d"/><text class="xs" x="${X(2) + 16}" y="${Y(0.25) + 5}">training (L_train = 4096)</text>` +
        `<rect x="${X(2)}" y="${Y(-0.25)}" width="12" height="4" fill="${C.q}"/><text class="xs" x="${X(2) + 16}" y="${Y(-0.25) + 5}">now (L_new = ${Lnew})</text>` });
    ro.innerHTML = `<b>${{ orig: "Original RoPE", pi: "Position Interpolation", ntk: "NTK-aware base", yarn: "YaRN (by parts)" }[meth]}</b>, s = ${s}<br>
      fastest pair speed: ×${fmt(tn[0] / th[0], 3)}<br>slowest pair speed: ×${fmt(tn[63] / th[63], 3)}<br>
      pairs seeing unseen angles: <b style="color:${unseen ? C.k : C.v}">${unseen}</b> of 64` + (meth === "ntk" ? `<br>new base = ${Math.round(base * Math.pow(s, dk / (dk - 2))).toLocaleString()}` : "");
  }
  r.oninput = draw;
  el.querySelectorAll(".meth button").forEach(b => b.onclick = () => { meth = b.dataset.m; el.querySelectorAll(".meth button").forEach(x => x.classList.toggle("on", x === b)); draw(); });
  draw();
};
