/* ReLU vs GELU vs SiLU explorer (§ Smooth activations). */
(function () {
  // erf via Abramowitz–Stegun 7.1.26 (max error 1.5e-7) — enough for a plot readout
  function erf(x) { const s = Math.sign(x); x = Math.abs(x); const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return s * y; }
  const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
  const F = {
    relu: { f: x => Math.max(0, x), d: x => x > 0 ? 1 : 0, color: "#85837d", name: "ReLU" },
    gelu: { f: x => x * Phi(x), d: x => Phi(x) + x * Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI), color: C.q, name: "GELU" },
    silu: { f: x => x * sigmoid(x), d: x => sigmoid(x) * (1 + x * (1 - sigmoid(x))), color: C.ffn, name: "SiLU" },
  };
  FIG.act = el => {
    el.innerHTML = `<div class="ctrl"><label>x = <input type="range" min="-4" max="3" step="0.05" value="-1"><output></output></label></div>
      <div class="panel"><div class="plot"></div><div class="readout"></div></div>`;
    const r = el.querySelector("input"), o = el.querySelector("output"), plot = el.querySelector(".plot"), ro = el.querySelector(".readout");
    function draw() {
      const x = +r.value; o.textContent = fmt(x, 2);
      plot.innerHTML = linePlot({ w: 520, h: 280, x: [-4, 3], y: [-0.6, 3], xticks: [-4, -3, -2, -1, 0, 1, 2, 3], yticks: [-0.5, 0, 1, 2, 3], xlabel: "x", ylabel: "output",
        series: Object.values(F).map(g => ({ f: g.f, color: g.color, wd: g === F.relu ? 2 : 2.4, dash: g === F.relu, n: 400 })),
        marks: Object.values(F).map(g => ({ x, y: g.f(x), color: g.color, tip: `${g.name}(${fmt(x, 2)}) = ${fmt(g.f(x), 4)}` })),
        extra: (X, Y) => `<line x1="${X(x)}" x2="${X(x)}" y1="${Y(-0.6)}" y2="${Y(3)}" stroke="${C.mut}" stroke-dasharray="3 3"/>` +
          Object.values(F).map((g, i) => `<rect x="${60 + i * 80}" y="18" width="12" height="4" fill="${g.color}"/><text class="sm" x="${76 + i * 80}" y="24">${g.name}</text>`).join("") });
      ro.innerHTML = `<b>x = ${fmt(x, 2)}</b><br>` + Object.values(F).map(g =>
        `<span style="color:${g.color};font-weight:700">${g.name}</span>: value ${fmt(g.f(x), 4)}, slope ${fmt(g.d(x), 3)}`).join("<br>") +
        `<br><span style="color:var(--muted)">The slope is what backpropagation multiplies by.<br>ReLU's slope is exactly 0 for every x &lt; 0.</span>`;
    }
    r.oninput = draw; draw();
  };
})();
