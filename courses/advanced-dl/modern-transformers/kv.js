/* KV-cache calculator (§ MQA and GQA). memory = 2 · layers · B · H_kv · T · d_k · bytes */
FIG.kvCalc = el => {
  const presets = {
    "70b": { name: "Llama-2-70B-like", L: 80, H: 64, Hkv: 8, dk: 128 },
    "8b": { name: "Llama-3-8B-like", L: 32, H: 32, Hkv: 8, dk: 128 },
    "base": { name: "2017 base model (decoder)", L: 6, H: 8, Hkv: 8, dk: 64 },
  };
  el.innerHTML = `<div class="ctrl">
    <label>model <select class="pre">${Object.entries(presets).map(([k, p]) => `<option value="${k}">${p.name}</option>`).join("")}</select></label>
    <label>context T <input class="t" type="range" min="10" max="17" step="1" value="12"><output class="to"></output></label>
    <label>batch B <input class="b" type="range" min="1" max="32" step="1" value="1"><output class="bo"></output></label>
    <label>bytes/number <select class="by"><option value="2">2 (16-bit)</option><option value="1">1 (8-bit)</option></select></label></div>
    <div class="panel"><div class="plot"></div><div class="readout"></div></div>`;
  const q = s => el.querySelector(s);
  function draw() {
    const p = presets[q(".pre").value], T = 2 ** +q(".t").value, B = +q(".b").value, by = +q(".by").value;
    q(".to").textContent = T >= 1024 ? (T / 1024) + "K" : T; q(".bo").textContent = B;
    const mem = hkv => 2 * p.L * B * hkv * T * p.dk * by;
    const rows = [["MHA", p.H, C.k], [`GQA (H_kv = ${p.Hkv})`, p.Hkv, C.q], ["MQA", 1, C.v]];
    const GiB = 2 ** 30, vals = rows.map(r => mem(r[1]) / GiB), mx = Math.max(80e9 / GiB, ...vals) * 1.1;
    const W = 520, x0 = 150, bw = W - x0 - 70; let s = "";
    rows.forEach((r, i) => { const w = Math.max(1, bw * vals[i] / mx), y = 20 + i * 46;
      s += `<text class="sm b" x="${x0 - 10}" y="${y + 20}" text-anchor="end">${r[0]}</text><rect x="${x0}" y="${y}" width="${w}" height="28" rx="4" fill="${r[2]}" data-tip="${fmt(vals[i], 3)} GiB"/>
            <text class="sm" x="${x0 + w + 6}" y="${y + 19}">${vals[i] >= 1 ? fmt(vals[i], 2) + " GiB" : fmt(vals[i] * 1024, 1) + " MiB"}</text>`; });
    const gx = x0 + bw * (80e9 / GiB) / mx;
    s += `<line x1="${gx}" x2="${gx}" y1="10" y2="160" stroke="${C.ink}" stroke-dasharray="5 4"/><text class="xs" x="${gx}" y="172" text-anchor="middle">80 GB GPU</text>`;
    q(".plot").innerHTML = svg(W, 180, s);
    q(".readout").innerHTML = `<b>${p.name}</b><br>layers ${p.L} · H ${p.H} · H_kv ${p.Hkv} · d_k ${p.dk}<br>
      2 × ${p.L} × ${B} × H_kv × ${T.toLocaleString()} × ${p.dk} × ${by} bytes<br>
      per token (B = 1): MHA ${fmt(mem(p.H) / B / T / 1024, 1)} KiB, GQA ${fmt(mem(p.Hkv) / B / T / 1024, 1)} KiB<br>
      GQA saves a factor <b>G = ${p.H / p.Hkv}</b>`;
  }
  el.querySelectorAll("input,select").forEach(x => x.oninput = draw); draw();
};
