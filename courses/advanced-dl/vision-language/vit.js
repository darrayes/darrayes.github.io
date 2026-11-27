/* Vision Transformer demos (§95–§98): token cost, patchify, patch embedding, positions.
   Example numbers arrive from data.json through data-j attributes, so they match the text. */
(function () {
  const gray = v => { const t = Math.round(255 * v / 9); return `rgb(${t},${t},${t})`; };   // 0 = black … 9 = white (as in §94)
  const PCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"];
  const pcol = (n, N) => N <= 4 ? PCOL[n] : `hsl(${Math.round(360 * n / N)},62%,48%)`;
  const comma = n => Math.round(n).toLocaleString("en-US");
  const human = b => b >= 2 ** 30 ? fmt(b / 2 ** 30, 2) + " GiB" : b >= 2 ** 20 ? fmt(b / 2 ** 20, 2) + " MiB" : b >= 1024 ? fmt(b / 1024, 1) + " KiB" : b + " bytes";

  /* ---------- §96: how many tokens? pixels vs patches, at any resolution ---------- */
  FIG.tokenCost = el => {
    el.innerHTML = `<div class="ctrl">
      <label>image side <select class="res">${[32, 64, 224, 384, 512, 1024].map(v => `<option ${v === 224 ? "selected" : ""}>${v}</option>`).join("")}</select> px</label>
      <label>one token = <select class="unit"><option value="1">1 pixel</option><option value="8">8×8 patch</option><option value="14">14×14 patch</option><option value="16" selected>16×16 patch</option><option value="32">32×32 patch</option></select></label></div>
      <div class="panel"><div class="viz"></div><div class="readout"></div></div>`;
    const rs = el.querySelector(".res"), us = el.querySelector(".unit"), viz = el.querySelector(".viz"), ro = el.querySelector(".readout");
    function draw() {
      const R = +rs.value, P = +us.value;
      if (R % P) { ro.innerHTML = `<b>${R} is not divisible by ${P}.</b><br>The image would be resized first.`; viz.innerHTML = ""; return; }
      const side = R / P, T = side * side, T2 = T * T, pix = R * R;
      // draw the image with its patch grid (at most 64 lines drawn; finer grids are shaded)
      const W = 220; let s = `<rect x="0" y="0" width="${W}" height="${W}" fill="${C.soft}" stroke="${C.ink2}"/>`;
      if (side <= 64) for (let i = 1; i < side; i++) { const p = W * i / side; s += `<line x1="${p}" x2="${p}" y1="0" y2="${W}" stroke="${C.q}" stroke-width=".7" opacity=".7"/><line y1="${p}" y2="${p}" x1="0" x2="${W}" stroke="${C.q}" stroke-width=".7" opacity=".7"/>`; }
      else s += `<rect x="0" y="0" width="${W}" height="${W}" fill="${C.q}" opacity=".35"/><text class="sm" x="${W / 2}" y="${W / 2}" text-anchor="middle">${side}×${side} grid: too fine to draw</text>`;
      s += `<rect x="0" y="0" width="${W / side}" height="${W / side}" fill="${C.k}" opacity=".85"/>`;
      viz.innerHTML = svg(W + 2, W + 24, `<g transform="translate(1,1)">${s}</g><text class="xs" x="${W / 2}" y="${W + 18}" text-anchor="middle">orange = one token</text>`);
      const vsText = T2 / 4096 ** 2;
      ro.innerHTML = `<b>${R}×${R} image, ${P === 1 ? "pixel tokens" : P + "×" + P + " patches"}</b><br>
        tokens T = ${side} × ${side} = <b>${comma(T)}</b><br>
        numbers per token = ${P}·${P}·3 = ${comma(P * P * 3)}<br>
        scores per head T² = <b>${comma(T2)}</b><br>
        score memory per head (2 bytes each) = ${human(T2 * 2)}<br>
        vs a 4,096-token text: ${vsText >= 1 ? fmt(vsText, 1) + "× more" : fmt(1 / vsText, 1) + "× fewer"} scores<br>
        <span style="color:var(--muted)">tokens × numbers per token = ${comma(T * P * P * 3)} = ${R}·${R}·3 ✓</span>`;
    }
    rs.onchange = us.onchange = draw; draw();
  };

  /* ---------- §96: cut the 4×4 example into patches and flatten them ---------- */
  FIG.patchify = el => {
    const J = JSON.parse(el.dataset.j), I = J.img4, n = I.length;
    el.innerHTML = `<div class="ctrl"><span>patch size P =</span>${[1, 2, 4].map(p => `<label><input type="radio" name="pp${n}" value="${p}" ${p === 2 ? "checked" : ""}> ${p}</label>`).join("")}</div>
      <div class="panel"><div class="viz"></div><div class="rows"></div></div><div class="readout" style="margin-top:8px"></div>`;
    const viz = el.querySelector(".viz"), rows = el.querySelector(".rows"), ro = el.querySelector(".readout");
    let P = 2, sel = 1;
    function patches() { const out = []; for (let a = 0; a < n; a += P) for (let b = 0; b < n; b += P) { const v = []; for (let u = a; u < a + P; u++) for (let w = b; w < b + P; w++) v.push(I[u][w]); out.push({ a, b, v }); } return out; }
    function draw() {
      const ps = patches(), N = ps.length, cs = 46; sel = Math.min(sel, N - 1);
      let s = "";
      for (let u = 0; u < n; u++) for (let w = 0; w < n; w++) { const v = I[u][w];
        s += `<rect x="${w * cs}" y="${u * cs}" width="${cs}" height="${cs}" fill="${gray(v)}" stroke="#ddd"/><text x="${w * cs + cs / 2}" y="${u * cs + cs / 2 + 5}" text-anchor="middle" style="font-size:15px;fill:${v < 5 ? "#fff" : C.ink}">${v}</text>`; }
      ps.forEach((p, k) => { s += `<rect class="pt" data-k="${k}" x="${p.b * cs + 2}" y="${p.a * cs + 2}" width="${P * cs - 4}" height="${P * cs - 4}" fill="${k === sel ? pcol(k, N) : "transparent"}" fill-opacity=".18" stroke="${pcol(k, N)}" stroke-width="${k === sel ? 4 : 2}" rx="3" style="cursor:pointer"/>`; });
      viz.innerHTML = svg(n * cs + 2, n * cs + 2, `<g transform="translate(1,1)">${s}</g>`);
      let t = `<div class="mt" style="font:600 12px sans-serif;color:var(--ink2);margin-bottom:4px">patch matrix X<sub>p</sub> · ${N} × ${P * P}</div><table class="pm">`;
      ps.forEach((p, k) => { t += `<tr data-k="${k}" class="${k === sel ? "on" : ""}" style="cursor:pointer"><th style="color:${pcol(k, N)}">patch ${k + 1}</th>${p.v.map(x => `<td style="background:${gray(x)};color:${x < 5 ? "#fff" : C.ink}">${x}</td>`).join("")}</tr>`; });
      rows.innerHTML = t + "</table>";
      const p = ps[sel];
      ro.innerHTML = `<b>P = ${P}</b>: N = (4/${P})·(4/${P}) = <b>${N}</b> token${N > 1 ? "s" : ""}, each with P²C = ${P}·${P}·1 = <b>${P * P}</b> number${P > 1 ? "s" : ""}. ${N} × ${P * P} = 16 = all pixels, none lost.<br>
        Patch ${sel + 1} covers rows ${p.a + 1}–${p.a + P}, columns ${p.b + 1}–${p.b + P}, read row by row: [${p.v.join(", ")}].
        ${P === 1 ? "<br><span style='color:var(--muted)'>P = 1 is the pixel-token case of §95: many tokens, one number each.</span>" : P === 4 ? "<br><span style='color:var(--muted)'>P = 4: one token holds the whole image, so there is nothing left to attend to.</span>" : ""}`;
      el.querySelectorAll("[data-k]").forEach(e => e.onclick = () => { sel = +e.dataset.k; draw(); });
    }
    el.querySelectorAll("input[type=radio]").forEach(r => r.onchange = () => { P = +r.value; draw(); });
    draw();
  };

  /* ---------- §97: patch embedding, one patch at a time, with editable pixels ---------- */
  FIG.patchEmbed = el => {
    const J = JSON.parse(el.dataset.j), W = J.Wp, X0 = J.Xp;
    const names = ["brightness", "left − right", "top − bottom"], pos = ["top-left", "top-right", "bottom-left", "bottom-right"];
    el.innerHTML = `<div class="ctrl"><span>start from</span>${X0.map((_, k) => `<button type="button" data-k="${k}">patch ${k + 1}</button>`).join("")}</div>
      <div class="ctrl sl">${pos.map((p, i) => `<label>${p} <input type="range" min="0" max="9" step="1" data-i="${i}"><output></output></label>`).join("")}</div>
      <div class="panel"><div class="viz"></div><div class="readout"></div></div>`;
    const sl = [...el.querySelectorAll(".sl input")], viz = el.querySelector(".viz"), ro = el.querySelector(".readout");
    let x = X0[1].slice();
    function set(v) { x = v.slice(); sl.forEach((s, i) => s.value = x[i]); draw(); }
    function draw() {
      sl.forEach((s, i) => { x[i] = +s.value; s.nextElementSibling.textContent = x[i]; });
      const z = [0, 1, 2].map(j => x.reduce((a, xi, i) => a + xi * W[i][j], 0));
      const cs = 54; let s = "";
      x.forEach((v, i) => { const r_ = i >> 1, c_ = i & 1; s += `<rect x="${c_ * cs}" y="${r_ * cs}" width="${cs}" height="${cs}" fill="${gray(v)}" stroke="#ccc"/><text x="${c_ * cs + cs / 2}" y="${r_ * cs + cs / 2 + 5}" text-anchor="middle" style="font-size:15px;fill:${v < 5 ? "#fff" : C.ink}">${v}</text>`; });
      // bars for z
      const bx = 130, mx = 9;
      z.forEach((v, j) => { const y = 6 + j * 30, w = Math.abs(v) / mx * 70;
        s += `<text class="sm" x="${bx}" y="${y + 14}">${names[j]}</text><line x1="${bx + 160}" x2="${bx + 160}" y1="${y}" y2="${y + 22}" stroke="${C.mut}"/>
          <rect x="${v >= 0 ? bx + 160 : bx + 160 - w}" y="${y + 3}" width="${w}" height="16" fill="${v >= 0 ? C.q : C.k}" rx="2"/><text class="sm b" x="${bx + 240}" y="${y + 15}" text-anchor="end">${fmt(v, 2)}</text>`; });
      viz.innerHTML = svg(bx + 248, 110, s);
      ro.innerHTML = `<b>x = [${x.join(", ")}]</b> → <b>z = [${z.map(v => fmt(v, 2)).join(", ")}]</b><br>` +
        [0, 1, 2].map(j => `${names[j]}: ${x.map((xi, i) => `${fmt(W[i][j], 2)}·${xi}`).join(" + ").replace(/\+ −/g, "− ")} = <b>${fmt(z[j], 2)}</b>`).join("<br>") +
        `<br><span style="color:var(--muted)">Same weights for every patch. Try a flat patch (all equal): both edge features become 0.</span>`;
    }
    el.querySelectorAll("button[data-k]").forEach(b => b.onclick = () => set(X0[+b.dataset.k]));
    sl.forEach(s => s.oninput = draw);
    set(X0[1]);
  };

  /* ---------- §98: move patches around, with and without position embeddings ---------- */
  FIG.posSwap = el => {
    const J = JSON.parse(el.dataset.j), Z = J.Z, E = J.Epos, Xp = J.Xp;
    el.innerHTML = `<div class="ctrl"><button type="button" class="swp">swap slots 1 ↔ 2</button><button type="button" class="sh">shuffle</button><button type="button" class="rs">reset</button>
      <label><input type="checkbox" class="pe" checked> add position embeddings</label></div>
      <div class="panel"><div class="viz"></div><div class="tab"></div></div><div class="readout" style="margin-top:8px"></div>`;
    const viz = el.querySelector(".viz"), tab = el.querySelector(".tab"), ro = el.querySelector(".readout"), pe = el.querySelector(".pe");
    let perm = [0, 1, 2, 3];   // perm[slot] = which original patch sits in that slot
    const row = (slot) => Z[perm[slot]].map((v, j) => v + (pe.checked ? E[slot][j] : 0));
    function draw() {
      const cs = 46; let s = "";
      perm.forEach((pk, slot) => { const a = (slot >> 1) * 2, b = (slot & 1) * 2;
        Xp[pk].forEach((v, i) => { const u = a + (i >> 1), w = b + (i & 1); s += `<rect x="${w * cs}" y="${u * cs}" width="${cs}" height="${cs}" fill="${gray(v)}"/>`; });
        s += `<rect x="${b * cs + 1.5}" y="${a * cs + 1.5}" width="${2 * cs - 3}" height="${2 * cs - 3}" fill="none" stroke="${PCOL[pk]}" stroke-width="3" rx="3"/><rect x="${b * cs + 3}" y="${a * cs + 3}" width="38" height="15" rx="3" fill="#fff" opacity=".9"/><text x="${b * cs + 6}" y="${a * cs + 14}" style="font-size:11px;fill:${PCOL[pk]};font-weight:700">slot ${slot + 1}</text>`; });
      viz.innerHTML = svg(4 * cs + 2, 4 * cs + 2, `<g transform="translate(1,1)">${s}</g>`);
      const rows = perm.map((_, slot) => row(slot)), base = [0, 1, 2, 3].map(slot => Z[slot].map((v, j) => v + (pe.checked ? E[slot][j] : 0)));
      let t = `<table class="pm"><tr><th>slot</th><th>holds</th><th colspan="3">${pe.checked ? "X = Z + E<sub>pos</sub>" : "Z only (no positions)"}</th></tr>`;
      rows.forEach((r_, slot) => { t += `<tr><th>${slot + 1}</th><td style="color:${PCOL[perm[slot]]};font-weight:700">patch ${perm[slot] + 1}</td>${r_.map(v => `<td>${fmt(v, 2)}</td>`).join("")}</tr>`; });
      tab.innerHTML = t + "</table>";
      const key = r_ => r_.map(v => fmt(v, 2)).join(",");
      const sameSet = JSON.stringify(rows.map(key).sort()) === JSON.stringify(base.map(key).sort());
      const moved = perm.some((p, i) => p !== i);
      ro.innerHTML = !moved ? `Original arrangement. Move some patches to compare.` :
        sameSet ? `<b>The encoder receives the same 4 vectors, only in a different order.</b> Attention would give the same outputs, reordered (§10): it cannot tell the two images apart.` :
        `<b>The vectors themselves changed</b>, because each patch picked up its new slot's position vector. The encoder can now tell the rearranged image from the original.`;
    }
    el.querySelector(".swp").onclick = () => { [perm[0], perm[1]] = [perm[1], perm[0]]; draw(); };
    el.querySelector(".sh").onclick = () => { let p; do { p = perm.slice().sort(() => Math.random() - .5); } while (p.join() === perm.join()); perm = p; draw(); };
    el.querySelector(".rs").onclick = () => { perm = [0, 1, 2, 3]; draw(); };
    pe.onchange = draw; draw();
  };
  /* ---------- §100: one attention head, seen from the [CLS] query ---------- */
  FIG.clsAttn = el => {
    const J = JSON.parse(el.dataset.j), X = J.Xc, WV = J.cls_W.WV, q0 = J.cls_W.WQ.map((r, i) => r[i]);
    const feats = ["brightness", "left − right", "top − bottom"], toks = ["[CLS]", "patch 1 · flat", "patch 2 · vertical edge", "patch 3 · horizontal edge", "patch 4 · mixed"];
    el.innerHTML = `<div class="ctrl"><span>[CLS] query weights (diagonal of W<sub>Q</sub>):</span></div>
      <div class="ctrl sl">${feats.map((f, i) => `<label>${f} <input type="range" min="-1" max="1" step="0.1" value="${q0[i]}"><output></output></label>`).join("")}<button type="button" class="rs">reset</button></div>
      <div class="panel"><div class="tab"></div><div class="readout"></div></div>`;
    const sl = [...el.querySelectorAll(".sl input")], tab = el.querySelector(".tab"), ro = el.querySelector(".readout");
    function draw() {
      const w = sl.map(x => +x.value); sl.forEach((x, i) => x.nextElementSibling.textContent = fmt(w[i], 1));
      const q = X[0].map((v, i) => v * w[i]);                       // x_cls · diag(w)
      const sc = X.map(k => k.reduce((a, v, i) => a + v * q[i], 0) / Math.sqrt(3));   // W_K = I
      const a = softmax(sc), V = X.map(r => r.map((v, i) => v * WV[i][i]));
      const o = [0, 1, 2].map(j => V.reduce((s, r, t) => s + a[t] * r[j], 0));
      let t = `<table class="pm"><tr><th>token</th><th>score</th><th>weight</th><th style="min-width:130px"></th></tr>`;
      toks.forEach((n, i) => { t += `<tr><th style="text-align:left">${n}</th><td>${fmt(sc[i], 3)}</td><td>${fmt(a[i], 3)}</td><td style="text-align:left"><span style="display:inline-block;height:10px;border-radius:2px;background:${i ? C.q : C.dec};width:${Math.round(120 * a[i])}px"></span></td></tr>`; });
      tab.innerHTML = t + "</table>";
      ro.innerHTML = `<b>q = [${q.map(v => fmt(v, 2)).join(", ")}]</b><br>scores = q·k / √3, weights = softmax(scores)<br><b>new [CLS] = Σ weight · v = [${o.map(v => fmt(v, 4)).join(", ")}]</b><br><span style="color:var(--muted)">Raise “brightness” and the bright patches win;<br>set all three to 0 and the weights become equal (1/5 each).</span>`;
    }
    sl.forEach(x => x.oninput = draw);
    el.querySelector(".rs").onclick = () => { sl.forEach((x, i) => x.value = q0[i]); draw(); };
    draw();
  };
  /* ---------- Part V walkthrough: the whole tiny ViT, recomputed live ---------- */
  FIG.tinyVit = el => {
    const W = JSON.parse(el.dataset.j).w_all;
    const erf = x => { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);   // A&S 7.1.26, |error| < 1.5e-7
      return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); };
    const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));
    const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
    const ln = r => { const m = r.reduce((a, b) => a + b, 0) / r.length, v = r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length; return r.map(x => (x - m) / Math.sqrt(v + 1e-5)); };
    const cols = (A, a, b) => A.map(r => r.slice(a, b)), T = A => A[0].map((_, j) => A.map(r => r[j]));
    let img = W.img.map(r => r.slice()), order = [0, 1, 2, 3];
    el.innerHTML = `<div class="ctrl"><label><input type="checkbox" class="pe" checked> position embeddings</label>
      <button type="button" class="sh">swap patch slots 1↔2 and 3↔4</button><button type="button" class="rs">reset</button></div>
      <div class="panel"><div class="viz"></div><div class="out"></div></div><div class="readout" style="margin-top:8px"></div>`;
    const viz = el.querySelector(".viz"), out = el.querySelector(".out"), ro = el.querySelector(".readout"), pe = el.querySelector(".pe");
    function forward() {
      const Xp = []; for (const a of [0, 4]) for (const b of [0, 4]) { const v = []; for (let u = a; u < a + 4; u++) for (let w = b; w < b + 4; w++) v.push(img[u][w] / 9); Xp.push(v); }
      const Z = mm(Xp, W.Wp), Zo = order.map(k => Z[k]);
      const X = [W.cls, ...Zo].map((r, i) => r.map((v, j) => v + (pe.checked ? W.Epos[i][j] : 0)));
      const Xn = X.map(ln), Q = mm(Xn, W.WQ), K = mm(Xn, W.WK), V = mm(Xn, W.WV);
      const A = [0, 1].map(h => mm(cols(Q, 2 * h, 2 * h + 2), T(cols(K, 2 * h, 2 * h + 2))).map(r => softmax(r.map(x => x / Math.SQRT2))));
      const O = [0, 1].map(h => mm(A[h], cols(V, 2 * h, 2 * h + 2)));
      const Oc = O[0].map((r, i) => [...r, ...O[1][i]]), X1 = mm(Oc, W.WO).map((r, i) => r.map((v, j) => v + X[i][j]));
      const Hh = mm(X1.map(ln), W.W1).map(r => r.map(gelu)), X2 = mm(Hh, W.W2).map((r, i) => r.map((v, j) => v + X1[i][j]));
      const h = ln(X2[0]), p = softmax(mm([h], W.Wh)[0]);
      return { Z, A, p, y: X2[0] };
    }
    function draw() {
      const R = forward(), cs = 30; let s = "";
      for (let u = 0; u < 8; u++) for (let w = 0; w < 8; w++) { const v = img[u][w];
        s += `<rect data-u="${u}" data-w="${w}" x="${w * cs}" y="${u * cs}" width="${cs}" height="${cs}" fill="${gray(v)}" stroke="#bbb" stroke-width=".5" style="cursor:pointer"/><text x="${w * cs + cs / 2}" y="${u * cs + cs / 2 + 4}" text-anchor="middle" style="font-size:11px;pointer-events:none;fill:${v < 5 ? "#fff" : C.ink}">${v}</text>`; }
      [0, 1, 2, 3].forEach(k => { const a = (k >> 1) * 4, b = (k & 1) * 4, slot = order.indexOf(k) + 1;
        s += `<rect x="${b * cs + 1.5}" y="${a * cs + 1.5}" width="${4 * cs - 3}" height="${4 * cs - 3}" fill="none" stroke="${PCOL[k]}" stroke-width="3" rx="3" style="pointer-events:none"/><rect x="${b * cs + 3}" y="${a * cs + 3}" width="104" height="15" rx="3" fill="#fff" opacity=".9" style="pointer-events:none"/><text x="${b * cs + 6}" y="${a * cs + 14}" style="font-size:11px;font-weight:700;fill:${PCOL[k]};pointer-events:none">patch ${k + 1} · slot ${slot}</text>`; });
      viz.innerHTML = svg(8 * cs + 2, 8 * cs + 2, `<g transform="translate(1,1)">${s}</g>`);
      const bar = (v, c) => `<span style="display:inline-block;height:9px;border-radius:2px;background:${c};width:${Math.round(90 * v)}px"></span>`;
      let t = `<table class="pm"><tr><th>token</th><th>head 1</th><th>head 2</th></tr>`;
      ["[CLS]", ...order.map(k => "patch " + (k + 1))].forEach((n, i) => { t += `<tr><th style="text-align:left">${n}</th><td style="text-align:left">${bar(R.A[0][0][i], C.q)} ${fmt(R.A[0][0][i], 3)}</td><td style="text-align:left">${bar(R.A[1][0][i], C.k)} ${fmt(R.A[1][0][i], 3)}</td></tr>`; });
      t += `</table><div style="font:600 12px sans-serif;color:var(--ink2);margin:10px 0 4px">class probabilities</div><table class="pm">` +
        ["A", "B", "C"].map((c, i) => `<tr><th>class ${c}</th><td style="text-align:left">${bar(R.p[i], C.dec)} ${fmt(R.p[i], 4)}</td></tr>`).join("") + `</table>`;
      out.innerHTML = `<div style="font:600 12px sans-serif;color:var(--ink2);margin-bottom:4px">where [CLS] looks (row 0 of each head)</div>` + t;
      const same = JSON.stringify(img) === JSON.stringify(W.img) && order.join() === "0,1,2,3" && pe.checked;
      const match = R.p.every((v, i) => Math.abs(v - W.p[i]) < 1e-6);
      ro.innerHTML = `patch tokens Z: ${R.Z.map((r, k) => `p${k + 1} [${r.map(v => fmt(v, 3)).join(", ")}]`).join(" · ")}<br>
        [CLS] output: [${R.y.map(v => fmt(v, 4)).join(", ")}]` +
        (same ? `<br><b>${match ? "✓ matches the walkthrough exactly" : "✗ does not match the walkthrough"}</b>` : "");
      viz.querySelectorAll("rect[data-u]").forEach(r => r.onclick = () => { const u = +r.dataset.u, w = +r.dataset.w; img[u][w] = (img[u][w] + 3) % 10; draw(); });
    }
    pe.onchange = draw;
    el.querySelector(".sh").onclick = () => { order = order.join() === "0,1,2,3" ? [1, 0, 3, 2] : [0, 1, 2, 3]; draw(); };
    el.querySelector(".rs").onclick = () => { img = W.img.map(r => r.slice()); order = [0, 1, 2, 3]; pe.checked = true; draw(); };
    draw();
  };
  /* ---------- §114: dot product vs cosine similarity in the plane ---------- */
  FIG.cosine = el => {
    const f = [3, 4], pres = { g1: [6, 8], g2: [4, -3], g3: [1, 1], g4: [30, 0] };
    el.innerHTML = `<div class="ctrl"><span>load</span>${Object.keys(pres).map(k => `<button type="button" data-k="${k}">${k}</button>`).join("")}</div>
      <div class="ctrl"><label>direction of g <input type="range" class="ang" min="-180" max="180" step="0.01"><output></output></label>
      <label>length of g <input type="range" class="len" min="0.5" max="30" step="0.01"><output></output></label></div>
      <div class="panel"><div class="viz"></div><div class="readout"></div></div>`;
    const ang = el.querySelector(".ang"), len = el.querySelector(".len"), viz = el.querySelector(".viz"), ro = el.querySelector(".readout");
    let g = [30, 0];
    function load(v) { g = v.slice(); ang.value = Math.atan2(g[1], g[0]) * 180 / Math.PI; len.value = Math.hypot(...g); draw(); }
    function fromSliders() { const a = +ang.value * Math.PI / 180, L = +len.value; g = [L * Math.cos(a), L * Math.sin(a)]; draw(); }
    function draw() {
      const L = Math.hypot(...g);
      ang.nextElementSibling.textContent = fmt(Math.atan2(g[1], g[0]) * 180 / Math.PI, 1) + "°"; len.nextElementSibling.textContent = fmt(L, 2);
      const dot = f[0] * g[0] + f[1] * g[1], cos = dot / (5 * L), th = Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
      const W = 300, c = W / 2, sc = (c - 14) / Math.max(5, L), U = 60;   // the longer vector reaches near the edge; unit circle drawn with radius U px
      const P = v => [c + v[0] * sc, c - v[1] * sc];
      let s = `<rect width="${W}" height="${W}" fill="${C.soft}" rx="8"/><line x1="0" x2="${W}" y1="${c}" y2="${c}" stroke="${C.line}"/><line y1="0" y2="${W}" x1="${c}" x2="${c}" stroke="${C.line}"/>`;
      s += `<circle cx="${c}" cy="${c}" r="${U}" fill="none" stroke="${C.mut}" stroke-dasharray="3 3"/><text x="${c + U * .72 + 4}" y="${c - U * .72 - 4}" style="font-size:10px;fill:${C.mut}">unit circle (scaled)</text>`;
      const arrow = (v, col, wd, mk) => { const [x, y] = P(v); return `<line x1="${c}" y1="${c}" x2="${x}" y2="${y}" stroke="${col}" stroke-width="${wd}" marker-end="url(#${mk})"/>`; };
      const unit = (v, col) => { const n = Math.hypot(...v); return `<circle cx="${c + U * v[0] / n}" cy="${c - U * v[1] / n}" r="4.5" fill="${col}"/>`; };
      s += arrow(f, C.q, 2.2, "ahq") + arrow(g, C.k, 2.2, "ahk") + unit(f, C.q) + unit(g, C.k);
      const [fx, fy] = P(f), [gx, gy] = P(g);
      s += `<text x="${fx + 6}" y="${fy - 4}" style="font-size:13px;font-weight:700;fill:${C.q}">f</text><text x="${gx + 6}" y="${gy - 4}" style="font-size:13px;font-weight:700;fill:${C.k}">g</text>`;
      viz.innerHTML = svg(W, W, s);
      ro.innerHTML = `f = [3, 4], ‖f‖ = 5<br>g = [${fmt(g[0], 2)}, ${fmt(g[1], 2)}], ‖g‖ = ${fmt(L, 2)}<br>
        <b>dot f·g = ${fmt(dot, 2)}</b><br><b>cosine = ${fmt(cos, 4)}</b> (angle ${fmt(th, 1)}°)<br>
        <span style="color:var(--muted)">Drag the length: the dot product grows with it,<br>the cosine does not change.<br>The dots on the circle are the unit vectors f̂ and ĝ;<br>their dot product is the cosine.</span>`;
    }
    el.querySelectorAll("button[data-k]").forEach(b => b.onclick = () => load(pres[b.dataset.k]));
    ang.oninput = len.oninput = fromSliders; load(pres.g4);
  };
  /* ---------- §116: the shared space, search and zero-shot labels ---------- */
  FIG.sharedSpace = el => {
    const J = JSON.parse(el.dataset.j).de_all, names = ["dog", "cat", "car"];
    const caps = ["a dog running on grass", "a cat on a sofa", "a red car on a road"];
    el.innerHTML = `<div class="ctrl"><label>direction of the new image f <input type="range" class="ang" min="-180" max="180" step="0.1"><output></output></label>
      <label>softmax scale <select class="sc"><option>1</option><option selected>10</option><option>100</option></select></label><button type="button" class="rs">reset</button></div>
      <div class="panel"><div class="viz"></div><div class="readout"></div></div>`;
    const ang = el.querySelector(".ang"), sc = el.querySelector(".sc"), viz = el.querySelector(".viz"), ro = el.querySelector(".readout");
    const a0 = Math.atan2(J.f[1], J.f[0]) * 180 / Math.PI; let f = J.f.slice();
    function draw() {
      ang.nextElementSibling.textContent = fmt(Math.atan2(f[1], f[0]) * 180 / Math.PI, 1) + "°";
      const W = 320, c = W / 2, R = 108, P = (v, r = R) => [c + r * v[0], c - r * v[1]];
      let s = `<rect width="${W}" height="${W}" rx="8" fill="${C.soft}"/><circle cx="${c}" cy="${c}" r="${R}" fill="none" stroke="${C.mut}" stroke-dasharray="3 3"/>`;
      const dot = (v, col, hollow) => { const [x, y] = P(v); return `<circle cx="${x}" cy="${y}" r="5.5" fill="${hollow ? "#fff" : col}" stroke="${col}" stroke-width="2"/>`; };
      J.Fh.forEach(v => s += dot(v, C.q)); J.Gh.forEach(v => s += dot(v, C.k)); J.Ph.forEach(v => s += dot(v, C.v, true));
      names.forEach((n, i) => { const m = [0, 1].map(k => J.Fh[i][k] + J.Gh[i][k] + J.Ph[i][k]), L = Math.hypot(...m), u = m.map(x => x / L), [x, y] = P(u, R + 24);
        s += `<text x="${x}" y="${y + 4}" text-anchor="middle" style="font-size:12.5px;font-weight:700;fill:${C.ink}">${n}</text>`; });
      const [fx, fy] = P(f), [mx, my] = P(f, R * 0.5);
      s += `<line x1="${c}" y1="${c}" x2="${fx}" y2="${fy}" stroke="${C.dec}" stroke-width="2.5" marker-end="url(#ahd)"/><text x="${mx}" y="${my + (f[1] >= 0 ? 16 : -8)}" text-anchor="middle" style="font-size:12px;font-weight:700;fill:${C.dec}">f</text>`;
      s += `<circle cx="14" cy="${W - 40}" r="5" fill="${C.q}"/><text x="24" y="${W - 36}" style="font-size:11px">photo</text><circle cx="72" cy="${W - 40}" r="5" fill="${C.k}"/><text x="82" y="${W - 36}" style="font-size:11px">caption</text><circle cx="140" cy="${W - 40}" r="5" fill="#fff" stroke="${C.v}" stroke-width="2"/><text x="150" y="${W - 36}" style="font-size:11px">prompt</text><text x="14" y="${W - 16}" style="font-size:11px;fill:${C.dec};font-weight:700">→ new image f</text>`;
      viz.innerHTML = svg(W, W, s);
      const cs = J.Ph.map(g => g[0] * f[0] + g[1] * f[1]), p = softmax(cs.map(x => x * +sc.value)), best = cs.indexOf(Math.max(...cs));
      ro.innerHTML = `<b>f = [${f.map(v => fmt(v, 4)).join(", ")}]</b><br>` +
        names.map((n, i) => `“a photo of a ${n}”: cosine ${fmt(cs[i], 4)} → p = ${fmt(p[i], 4)}`).join("<br>") +
        `<br><b>zero-shot label: ${names[best]}</b><br><span style="color:var(--muted)">Turn f around the circle.<br>The scale changes the probabilities, never the pick.</span>`;
    }
    ang.oninput = () => { const a = +ang.value * Math.PI / 180; f = [Math.cos(a), Math.sin(a)]; draw(); }; sc.onchange = draw;
    el.querySelector(".rs").onclick = () => { f = J.f.slice(); ang.value = a0; sc.value = "10"; draw(); };
    ang.value = a0; draw();
  };
  /* ---------- §118: the LLM's input sequence and its causal mask ---------- */
  FIG.prefixMask = el => {
    const txt = ["How", "many", "dogs", "?"];
    el.innerHTML = `<div class="ctrl"><label>visual tokens drawn N<sub>v</sub> <input type="range" class="nv" min="1" max="8" step="1" value="4"><output></output></label>
      <label>real image size <select class="real"><option value="64">64 visual tokens</option><option value="576" selected>576 (LLaVA-1.5)</option><option value="2880">2,880 (5 tiles × 576)</option></select></label></div>
      <div class="panel"><div class="viz"></div><div class="readout"></div></div>`;
    const nv = el.querySelector(".nv"), real = el.querySelector(".real"), viz = el.querySelector(".viz"), ro = el.querySelector(".readout");
    let sel = null;
    function draw() {
      const n = +nv.value; nv.nextElementSibling.textContent = n;
      const labs = [...Array(n)].map((_, i) => "v" + (i + 1)).concat(txt), L = labs.length; if (sel === null || sel >= L) sel = L - 1;
      const cs = Math.min(30, Math.floor(300 / L)), off = 46; let s = "";
      labs.forEach((l, i) => { const vis = i < n, col = vis ? C.ffn : C.q;
        s += `<text x="${off + i * cs + cs / 2}" y="${off - 6}" text-anchor="middle" style="font-size:10px;fill:${col}">${l}</text>`;
        s += `<text x="${off - 4}" y="${off + i * cs + cs / 2 + 4}" text-anchor="end" style="font-size:10px;fill:${col};font-weight:${i === sel ? 700 : 400}">${l}</text>`;
        labs.forEach((_, j) => { const ok = j <= i, onRow = i === sel;
          const fill = !ok ? "url(#hatch)" : (j < n ? (onRow ? C.ffn : "#f6e7c4") : (onRow ? C.q : C.q + "33"));
          s += `<rect data-r="${i}" x="${off + j * cs}" y="${off + i * cs}" width="${cs - 1}" height="${cs - 1}" fill="${fill}" style="cursor:pointer"/>`; }); });
      viz.innerHTML = svg(off + L * cs + 4, off + L * cs + 4, s);
      viz.querySelectorAll("rect[data-r]").forEach(r => r.onclick = () => { sel = +r.dataset.r; draw(); });
      const seen = Math.min(sel + 1, n), R = +real.value, Tq = 20, Lr = R + Tq, tot = Lr * (Lr + 1) / 2, img = R * (R + 1) / 2 + Tq * R;
      ro.innerHTML = `<b>row “${labs[sel]}”</b> sees ${sel + 1} of ${L} tokens, ${seen} of them visual.<br>
        ${sel >= n ? "Every text row sees <b>all</b> visual tokens: the image comes first." : "A visual row sees only the visual tokens before it."}<br><br>
        <b>With ${R.toLocaleString("en-US")} visual tokens and a 20-token question:</b><br>
        input length = ${Lr.toLocaleString("en-US")} tokens<br>
        allowed (unmasked) scores per head = ${tot.toLocaleString("en-US")}<br>
        of which involve visual Keys: ${img.toLocaleString("en-US")} (${fmt(100 * img / tot, 1)}%)<br>
        <span style="color:var(--muted)">Click any row of the grid to select it.</span>`;
    }
    nv.oninput = real.onchange = draw; draw();
  };
  /* ---------- §119: the gate of a cross-attention layer ---------- */
  FIG.xattnGate = el => {
    const J = JSON.parse(el.dataset.j).xa_all, txt = ["How", "many", "dogs"];
    el.innerHTML = `<div class="ctrl"><label>gate α <input type="range" class="al" min="-3" max="3" step="0.05" value="0"><output></output></label><button type="button" class="z">α = 0</button><button type="button" class="h">α = 0.5</button></div>
      <div class="panel"><div class="viz"></div><div class="readout"></div></div>`;
    const al = el.querySelector(".al"), viz = el.querySelector(".viz"), ro = el.querySelector(".readout");
    function draw() {
      const a = +al.value, t = Math.tanh(a); al.nextElementSibling.textContent = fmt(a, 2);
      let h = `<div style="font:600 12px sans-serif;color:var(--ink2);margin-bottom:4px">text → image attention weights</div><table class="pm"><tr><th></th>${["v1", "v2", "v3", "v4"].map(v => `<th>${v}</th>`).join("")}</tr>`;
      J.A.forEach((r, i) => { h += `<tr><th>${txt[i]}</th>${r.map(v => `<td style="background:${colSeq(v, 0.5)};color:${v > 0.3 ? "#fff" : C.ink}">${fmt(v, 3)}</td>`).join("")}</tr>`; });
      viz.innerHTML = h + "</table>";
      ro.innerHTML = `<b>tanh(α) = ${fmt(t, 4)}</b><br>` + J.U.map((u, i) => `${txt[i]}: [${u.map((v, j) => fmt(v + t * J.O[i][j], 4)).join(", ")}]`).join("<br>") +
        `<br><span style="color:var(--muted)">${Math.abs(t) < 1e-9 ? "α = 0: the rows are exactly the LLM's own; the image has no effect." : "The image's contribution is scaled by tanh(α), at most ±1."}</span>`;
    }
    al.oninput = draw; el.querySelector(".z").onclick = () => { al.value = 0; draw(); }; el.querySelector(".h").onclick = () => { al.value = 0.5; draw(); };
    draw();
  };

  /* ---------- §120: prefix versus learned queries, in tokens and attention scores ---------- */
  FIG.resampleCost = el => {
    el.innerHTML = `<div class="ctrl"><label>visual tokens from the ViT N <select class="n">${[64, 256, 576, 1024, 2880].map(v => `<option ${v === 576 ? "selected" : ""}>${v}</option>`).join("")}</select></label>
      <label>learned queries N<sub>q</sub> <select class="k">${[16, 32, 64, 128].map(v => `<option ${v === 64 ? "selected" : ""}>${v}</option>`).join("")}</select></label>
      <label>question tokens T <input type="range" class="t" min="5" max="200" step="5" value="20"><output></output></label></div>
      <div class="panel"><div class="viz"></div><div class="readout"></div></div>`;
    const n = el.querySelector(".n"), k = el.querySelector(".k"), t = el.querySelector(".t"), viz = el.querySelector(".viz"), ro = el.querySelector(".readout");
    const c = x => Math.round(x).toLocaleString("en-US");
    function draw() {
      const N = +n.value, K = +k.value, T = +t.value; t.nextElementSibling.textContent = T;
      const Lp = N + T, Lq = K + T, sp = Lp * (Lp + 1) / 2, sq = Lq * (Lq + 1) / 2, sr = K * N;
      const bars = [["prefix: LLM scores", sp, C.ffn], ["learned queries: LLM scores", sq, C.q], ["learned queries: resampler scores", sr, C.v]], mx = Math.max(...bars.map(b => b[1]));
      viz.innerHTML = svg(330, 110, bars.map((b, i) => `<text x="0" y="${18 + i * 34}" style="font-size:11px">${b[0]}</text><rect x="0" y="${23 + i * 34}" width="${Math.max(1, 320 * b[1] / mx)}" height="9" rx="2" fill="${b[2]}"/>`).join(""));
      ro.innerHTML = `<b>Prefix (§118)</b>: LLM input ${c(Lp)} tokens → ${c(sp)} causal scores per head per layer.<br>
        <b>Learned queries</b>: LLM input ${c(Lq)} tokens → ${c(sq)} scores per head per layer,<br>plus ${c(sr)} scores per head in each resampler layer.<br>
        Visual tokens in the LLM: <b>${c(N)} → ${c(K)}</b> (${fmt(N / K, 1)}× fewer).<br>
        <span style="color:var(--muted)">The LLM has many layers and the resampler few, so the LLM side dominates.</span>`;
    }
    n.onchange = k.onchange = t.oninput = draw; draw();
  };
})();
