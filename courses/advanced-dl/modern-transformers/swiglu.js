/* SwiGLU demos (§ The gated FFN: SwiGLU):
   FIG.swGate   — the worked example made live: move x, watch each hidden unit's gate open or shut,
                  and see where the units sit on the map h = SiLU(a)·u (or ReLU(a) for the ungated FFN).
   FIG.swBudget — parameter budget: two-matrix FFN (8D²) versus three-matrix SwiGLU (3·D·d_ff). */
(function () {
  const silu = z => z * sigmoid(z), relu = z => Math.max(0, z);
  const matvec = (x, W) => W[0].map((_, j) => x.reduce((s, xi, i) => s + xi * W[i][j], 0));
  const AMB = C.ffn, UP = C.q, H = C.dec;

  FIG.swGate = el => {
    const W = JSON.parse(el.dataset.w), Wg = W.sw_Wg, Wu = W.sw_Wu, Wd = W.sw_Wd;
    const dff = Wg[0].length, D = Wg.length;
    el.innerHTML = `<div class="ctrl">
        <label>x₁ <input type="range" min="-3" max="3" step="0.1" value="1"><output></output></label>
        <label>x₂ <input type="range" min="-3" max="3" step="0.1" value="2"><output></output></label></div>
      <div class="ctrl"><button data-x="1,2">x = [1, 2] (worked example)</button><button data-x="-1,2">x = [−1, 2]</button><button data-x="2,1">x = [2, 1]</button>
        <span class="meth"><button data-m="sw" class="on">SwiGLU</button><button data-m="relu">ReLU FFN</button></span></div>
      <div class="panel"><div class="pipe"></div><div class="map"></div></div><div class="panel"><div class="readout"></div></div>`;
    const [r1, r2] = el.querySelectorAll("input[type=range]"), outs = el.querySelectorAll("output");
    const pipe = el.querySelector(".pipe"), map = el.querySelector(".map"), ro = el.querySelector(".readout");
    let mode = "sw";

    // a centred horizontal bar for value v in a cell [x0, x0+w], scale m = |v| that fills half the cell
    function bar(x0, w, y, v, m, color, label, dim) {
      const cx = x0 + w / 2, len = Math.max(-1, Math.min(1, v / m)) * (w / 2 - 4);
      return `<line x1="${cx}" x2="${cx}" y1="${y - 13}" y2="${y + 13}" stroke="${C.line}"/>` +
        `<rect x="${Math.min(cx, cx + len)}" y="${y - 9}" width="${Math.abs(len)}" height="18" rx="3" fill="${color}" opacity="${dim ? 0.18 : 0.85}" data-tip="${label} = ${fmt(v, 4)}"/>` +
        `<text class="sm b" x="${cx}" y="${y + 28}" text-anchor="middle" ${dim ? 'opacity=".35"' : ""}>${fmt(v, 3)}</text>`;
    }

    function draw() {
      const x = [+r1.value, +r2.value]; outs[0].textContent = fmt(x[0], 1); outs[1].textContent = fmt(x[1], 1);
      const a = matvec(x, Wg), u = matvec(x, Wu), g = a.map(mode === "sw" ? silu : relu);
      const h = mode === "sw" ? g.map((v, j) => v * u[j]) : g, y = matvec(h, Wd);
      const ySw = matvec(a.map((v, j) => silu(v) * u[j]), Wd), yRe = matvec(a.map(relu), Wd);

      // ---- pipeline: one row per hidden unit
      const cw = 112, L = 64, top = 46, rh = 62, Wd_ = L + 4 * cw + 8, Ht = top + dff * rh + 92;
      const heads = mode === "sw" ? ["gate input a", "SiLU(a)", "up u", "h = SiLU(a)·u"] : ["a = xW_gate", "ReLU(a)", "u (not used)", "h = ReLU(a)"];
      const cols = [C.ink2, AMB, UP, H];
      let s = heads.map((t, c) => `<text class="sm b" x="${L + c * cw + cw / 2}" y="18" text-anchor="middle" style="fill:${cols[c]}">${t}</text>`).join("");
      const m = Math.max(3, ...a.map(Math.abs), ...u.map(Math.abs), ...h.map(Math.abs));
      for (let j = 0; j < dff; j++) {
        const yy = top + j * rh + 14;
        const state = mode === "sw" ? (Math.abs(g[j]) < 0.05 ? "shut" : a[j] < 0 ? "leaks (sign flips)" : "open")
                                    : (a[j] > 0 ? "open" : "shut");
        s += `<text class="sm b" x="6" y="${yy - 2}">unit ${j + 1}</text><text class="xs" x="6" y="${yy + 12}" style="fill:${state === "open" ? C.v : state === "shut" ? C.mut : C.k}">${state}</text>`;
        s += bar(L, cw, yy, a[j], m, C.ink2, `a${j + 1}`);
        s += bar(L + cw, cw, yy, g[j], m, AMB, `${mode === "sw" ? "SiLU" : "ReLU"}(a${j + 1})`);
        s += bar(L + 2 * cw, cw, yy, u[j], m, UP, `u${j + 1}`, mode !== "sw");
        s += bar(L + 3 * cw, cw, yy, h[j], m, H, `h${j + 1}`);
      }
      const yb = top + dff * rh + 22;
      s += `<line x1="${L}" x2="${L + 4 * cw}" y1="${yb - 22}" y2="${yb - 22}" stroke="${C.line}"/>`;
      s += `<text class="sm b" x="6" y="${yb + 4}">y = hW_down</text>`;
      y.forEach((v, i) => s += bar(L + (1 + i) * cw + cw / 2, cw, yb + 10, v, Math.max(3, ...y.map(Math.abs)), C.ink, `y${i + 1}`) +
        `<text class="xs" x="${L + (1 + i) * cw + cw}" y="${yb - 8}" text-anchor="middle">y${"₁₂₃"[i]}</text>`);
      pipe.innerHTML = svg(Wd_, Ht, s);

      // ---- map of the hidden unit's output over (a, u)
      const R = 4, n = 40, P = 230, ox = 36, oy = 10, cell = P / n, X = v => ox + (v + R) / (2 * R) * P, Y = v => oy + (R - v) / (2 * R) * P;
      let t = "";
      for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) {
        const av = -R + (i + 0.5) * 2 * R / n, uv = R - (k + 0.5) * 2 * R / n;
        const hv = mode === "sw" ? silu(av) * uv : relu(av);
        t += `<rect x="${(ox + i * cell).toFixed(1)}" y="${(oy + k * cell).toFixed(1)}" width="${(cell + 0.4).toFixed(1)}" height="${(cell + 0.4).toFixed(1)}" fill="${colDiv(hv, 8)}"/>`;
      }
      t += `<line x1="${X(0)}" x2="${X(0)}" y1="${oy}" y2="${oy + P}" stroke="${C.ink2}" stroke-dasharray="3 3"/>`;
      if (mode === "sw") t += `<line x1="${ox}" x2="${ox + P}" y1="${Y(0)}" y2="${Y(0)}" stroke="${C.ink2}" stroke-dasharray="3 3"/>`;
      t += `<rect x="${ox}" y="${oy}" width="${P}" height="${P}" fill="none" stroke="${C.line}"/>`;
      [-4, -2, 0, 2, 4].forEach(v => t += `<text class="xs" x="${X(v)}" y="${oy + P + 14}" text-anchor="middle">${fmt(v)}</text><text class="xs" x="${ox - 5}" y="${Y(v) + 3}" text-anchor="end">${fmt(v)}</text>`);
      t += `<text class="sm" x="${ox + P / 2}" y="${oy + P + 30}" text-anchor="middle" style="fill:${AMB}">gate input a</text>`;
      t += `<text class="sm" transform="translate(10,${oy + P / 2}) rotate(-90)" text-anchor="middle" style="fill:${UP}">up value u${mode === "sw" ? "" : " (ignored)"}</text>`;
      for (let j = 0; j < dff; j++) {
        const cx = X(Math.max(-R, Math.min(R, a[j]))), cy = Y(Math.max(-R, Math.min(R, mode === "sw" ? u[j] : 0)));
        t += `<circle cx="${cx}" cy="${cy}" r="9" fill="#fff" stroke="${C.ink}" stroke-width="1.5" data-tip="unit ${j + 1}: a = ${fmt(a[j], 3)}, u = ${fmt(u[j], 3)}, h = ${fmt(h[j], 3)}"/>` +
             `<text class="xs b" x="${cx}" y="${cy + 3.5}" text-anchor="middle" pointer-events="none">${j + 1}</text>`;
      }
      t += `<text class="xs" x="${ox + P + 8}" y="${oy + 10}">h &gt; 0</text><rect x="${ox + P + 8}" y="${oy + 16}" width="14" height="10" fill="${DIV_POS}"/>`;
      t += `<text class="xs" x="${ox + P + 8}" y="${oy + 44}">h &lt; 0</text><rect x="${ox + P + 8}" y="${oy + 50}" width="14" height="10" fill="${DIV_NEG}"/>`;
      map.innerHTML = svg(ox + P + 52, oy + P + 38, t);

      ro.innerHTML = `<b>x = [${fmt(x[0], 1)}, ${fmt(x[1], 1)}]</b> &nbsp; SwiGLU y = [${ySw.map(v => fmt(v, 4)).join(", ")}] &nbsp;·&nbsp; ReLU FFN y = [${yRe.map(v => fmt(v, 4)).join(", ")}]<br>` +
        (mode === "sw"
          ? `Each unit's output is a <b>product</b>: the gate (amber) scales the up value (blue). On the map, the colour is h; the dashed lines a = 0 and u = 0 are where h = 0.`
          : `Ungated: each unit's output depends on a alone, so the map is constant along u. Every unit with a ≤ 0 is exactly 0 and passes no gradient to its weights.`);
    }
    [r1, r2].forEach(r => r.oninput = draw);
    el.querySelectorAll("button[data-x]").forEach(b => b.onclick = () => { const [p, q] = b.dataset.x.split(","); r1.value = p; r2.value = q; draw(); });
    el.querySelectorAll(".meth button").forEach(b => b.onclick = () => { mode = b.dataset.m; el.querySelectorAll(".meth button").forEach(z => z.classList.toggle("on", z === b)); draw(); });
    draw();
  };

  FIG.swBudget = el => {
    el.innerHTML = `<div class="ctrl"><label>D <select><option>768</option><option>2048</option><option selected>4096</option><option>8192</option></select></label>
        <label>d_ff / D <input type="range" min="1.5" max="4" step="0.01" value="2.6667"><output></output></label></div>
      <div class="ctrl"><button data-r="8/3">exact 8D/3</button><button data-r="r256">8D/3 rounded up to ×256</button><button data-r="3.5">Llama 3 8B (3.5D)</button><button data-r="4">4D (no shrink)</button></div>
      <div class="panel"><div class="bars"></div><div class="readout"></div></div>`;
    const sel = el.querySelector("select"), rg = el.querySelector("input"), out = el.querySelector("output"), bars = el.querySelector(".bars"), ro = el.querySelector(".readout");
    let dffFixed = null;                     // set when a preset pins an exact integer d_ff
    const n = v => Number.isInteger(v) ? v.toLocaleString("en-US") : (Math.abs(v - Math.round(v)) < 1e-6 ? Math.round(v).toLocaleString("en-US") : fmt(v, 2));
    function draw() {
      const D = +sel.value, dff = dffFixed !== null ? dffFixed : Math.round(+rg.value * D);
      out.textContent = fmt(dff / D, 3) + " → d_ff = " + n(dff);
      const std = 2 * D * 4 * D, sw = 3 * D * dff, mx = Math.max(std, sw, 1) * 1.04;
      const L = 110, Wb = 360, sc = v => v / mx * Wb;
      let s = "", x;
      s += `<text class="sm b" x="0" y="30">FFN (2017)</text><text class="xs" x="0" y="44">2 matrices</text>`; x = L;
      [["W₁", 4 * D * D], ["W₂", 4 * D * D]].forEach(([k, v]) => { s += `<rect x="${x}" y="18" width="${sc(v)}" height="30" fill="${C.ffn}" opacity=".55" stroke="#fff" data-tip="${k}: ${n(v)} parameters"/><text class="xs" x="${x + sc(v) / 2}" y="37" text-anchor="middle">${k}</text>`; x += sc(v); });
      s += `<text class="xs b" x="${x + 6}" y="37">${n(std)}</text>`;
      s += `<text class="sm b" x="0" y="90">SwiGLU</text><text class="xs" x="0" y="104">3 matrices</text>`; x = L;
      [["W_gate", AMB], ["W_up", UP], ["W_down", H]].forEach(([k, c]) => { const v = D * dff; s += `<rect x="${x}" y="78" width="${sc(v)}" height="30" fill="${c}" opacity=".6" stroke="#fff" data-tip="${k}: ${n(D)} × ${n(dff)} = ${n(v)}"/><text class="xs" x="${x + sc(v) / 2}" y="97" text-anchor="middle">${k}</text>`; x += sc(v); });
      s += `<text class="xs b" x="${x + 6}" y="97">${n(sw)}</text>`;
      s += `<line x1="${L + sc(std)}" x2="${L + sc(std)}" y1="10" y2="118" stroke="${C.ink}" stroke-dasharray="4 3"/>`;
      bars.innerHTML = svg(L + Wb + 90, 126, s);
      const ratio = sw / std, pct = 100 * (ratio - 1);
      ro.innerHTML = `<b>D = ${n(D)}, d_ff = ${n(dff)}</b><br>FFN (2017): 2 · D · 4D = <b>${n(std)}</b><br>SwiGLU: 3 · D · d_ff = <b>${n(sw)}</b><br>` +
        `ratio SwiGLU / FFN = <b style="color:${Math.abs(pct) < 0.5 ? C.v : C.k}">${fmt(ratio, 4)}</b> (${pct >= 0 ? "+" : ""}${fmt(pct, 2)}%)<br>` +
        `<span style="color:var(--muted)">Equal counts need d_ff = 8D/3 = ${fmt(8 * D / 3, 2)},<br>which is not a whole number unless 3 divides D.</span>`;
    }
    sel.onchange = () => { dffFixed = null; draw(); };
    rg.oninput = () => { dffFixed = null; draw(); };
    el.querySelectorAll("button[data-r]").forEach(b => b.onclick = () => {
      const D = +sel.value, r = b.dataset.r;
      dffFixed = r === "8/3" ? null : r === "r256" ? Math.ceil(8 * D / 3 / 256) * 256 : Math.round(+r * D);
      rg.value = r === "8/3" ? 8 / 3 : dffFixed / D;
      if (r === "8/3") { dffFixed = 8 * D / 3; }
      draw();
    });
    el.querySelector('button[data-r="8/3"]').click();   // start on the equal-count width
  };
})();
