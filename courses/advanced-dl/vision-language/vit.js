/* Vision Transformer demos (§95–§98): token cost, patchify, patch embedding, positions.
   Example numbers arrive from data.json through data-j attributes, so they match the text. */
(function () {
  const gray = v => { const t = Math.round(255 * (1 - v / 9)); return `rgb(${t},${t},${t})`; };   // 0 = white … 9 = black
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
        s += `<rect x="${w * cs}" y="${u * cs}" width="${cs}" height="${cs}" fill="${gray(v)}" stroke="#ddd"/><text x="${w * cs + cs / 2}" y="${u * cs + cs / 2 + 5}" text-anchor="middle" style="font-size:15px;fill:${v > 4 ? "#fff" : C.ink}">${v}</text>`; }
      ps.forEach((p, k) => { s += `<rect class="pt" data-k="${k}" x="${p.b * cs + 2}" y="${p.a * cs + 2}" width="${P * cs - 4}" height="${P * cs - 4}" fill="${k === sel ? pcol(k, N) : "transparent"}" fill-opacity=".18" stroke="${pcol(k, N)}" stroke-width="${k === sel ? 4 : 2}" rx="3" style="cursor:pointer"/>`; });
      viz.innerHTML = svg(n * cs + 2, n * cs + 2, `<g transform="translate(1,1)">${s}</g>`);
      let t = `<div class="mt" style="font:600 12px sans-serif;color:var(--ink2);margin-bottom:4px">patch matrix X<sub>p</sub> · ${N} × ${P * P}</div><table class="pm">`;
      ps.forEach((p, k) => { t += `<tr data-k="${k}" class="${k === sel ? "on" : ""}" style="cursor:pointer"><th style="color:${pcol(k, N)}">patch ${k + 1}</th>${p.v.map(x => `<td style="background:${gray(x)};color:${x > 4 ? "#fff" : C.ink}">${x}</td>`).join("")}</tr>`; });
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
      x.forEach((v, i) => { const r_ = i >> 1, c_ = i & 1; s += `<rect x="${c_ * cs}" y="${r_ * cs}" width="${cs}" height="${cs}" fill="${gray(v)}" stroke="#ccc"/><text x="${c_ * cs + cs / 2}" y="${r_ * cs + cs / 2 + 5}" text-anchor="middle" style="font-size:15px;fill:${v > 4 ? "#fff" : C.ink}">${v}</text>`; });
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
})();
