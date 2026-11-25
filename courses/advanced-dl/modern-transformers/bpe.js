/* BPE merge stepper (§ Byte-Pair Encoding). Same tie rule as compute.py:
   highest count wins; on a tie, the pair seen first (scanning words in order) wins. */
FIG.bpe = el => {
  el.innerHTML = `
  <div class="panel">
    <div style="flex:1 1 220px;min-width:200px">
      <label class="ui" style="font-size:12.5px;color:var(--ink2)">Corpus (word count)</label>
      <textarea rows="5" spellcheck="false">low 5\nlower 2\nnewest 6\nwidest 3</textarea>
      <div class="ctrl" style="justify-content:flex-start">
        <button class="step">Merge ▸</button><button class="run">Merge ×5</button><button class="reset">Reset</button>
      </div>
      <div class="readout"></div>
    </div>
    <div style="flex:2 1 320px;min-width:260px">
      <div class="words"></div>
      <div class="ui" style="font-size:12.5px;color:var(--ink2);margin-top:10px">Merges learned so far</div>
      <ol class="merges" style="font:13px 'SF Mono',Menlo,monospace;margin:4px 0;padding-left:22px"></ol>
    </div>
  </div>`;
  const ta = el.querySelector("textarea"), wordsEl = el.querySelector(".words"),
        mergesEl = el.querySelector(".merges"), out = el.querySelector(".readout");
  let words, merges, lastNew;
  function parse() {
    words = ta.value.split("\n").map(l => l.trim().split(/\s+/)).filter(a => a[0])
      .map(([w, c]) => ({ sym: [...w, "_"], c: Math.max(1, parseInt(c) || 1), w }));
    merges = []; lastNew = null;
  }
  function pairs() {
    const m = new Map();
    words.forEach(w => { for (let i = 0; i + 1 < w.sym.length; i++) { const k = w.sym[i] + "\u0001" + w.sym[i + 1]; m.set(k, (m.get(k) || 0) + w.c); } });
    return m;
  }
  function best() { let bk = null, bc = 0; pairs().forEach((c, k) => { if (c > bc) { bc = c; bk = k; } }); return bk ? { a: bk.split("\u0001")[0], b: bk.split("\u0001")[1], c: bc } : null; }
  function merge() {
    const p = best(); if (!p) return false;
    words.forEach(w => { const o = []; for (let i = 0; i < w.sym.length;) { if (i + 1 < w.sym.length && w.sym[i] === p.a && w.sym[i + 1] === p.b) { o.push(p.a + p.b); i += 2; } else { o.push(w.sym[i]); i++; } } w.sym = o; });
    merges.push(p); lastNew = p.a + p.b; return true;
  }
  function draw() {
    const nxt = best();
    wordsEl.innerHTML = words.map(w => {
      let h = "", i = 0;
      while (i < w.sym.length) {
        if (nxt && i + 1 < w.sym.length && w.sym[i] === nxt.a && w.sym[i + 1] === nxt.b) {
          h += `<span style="background:var(--k-soft);border-color:var(--k)">${esc(w.sym[i])}</span><span style="background:var(--k-soft);border-color:var(--k)">${esc(w.sym[i + 1])}</span>`; i += 2;
        } else { h += `<span class="${w.sym[i] === lastNew ? "new" : ""}" style="${w.sym[i] === lastNew ? "background:var(--v-soft);border-color:var(--v)" : ""}">${esc(w.sym[i])}</span>`; i++; }
      }
      return `<div class="wrow"><span class="cnt">×${w.c}</span><div class="toks">${h}</div></div>`;
    }).join("");
    mergesEl.innerHTML = merges.map(m => `<li>${esc(m.a)} + ${esc(m.b)} → <b>${esc(m.a + m.b)}</b> <span style="color:var(--muted)">(${m.c})</span></li>`).join("") || `<li style="list-style:none;color:var(--muted)">none yet</li>`;
    const vocab = new Set(); words.forEach(w => [...w.w, "_"].forEach(ch => vocab.add(ch))); merges.forEach(m => vocab.add(m.a + m.b));
    const total = words.reduce((s, w) => s + w.c * w.sym.length, 0), chars = words.reduce((s, w) => s + w.c * (w.w.length + 1), 0);
    out.innerHTML = `vocabulary: <b>${vocab.size}</b> symbols<br>corpus length: <b>${total}</b> symbols (from ${chars})<br>` +
      (nxt ? `next merge: <b>${esc(nxt.a)} + ${esc(nxt.b)}</b> (count ${nxt.c})` : `<b>nothing left to merge</b>`);
  }
  el.querySelector(".step").onclick = () => { merge(); draw(); };
  el.querySelector(".run").onclick = () => { for (let k = 0; k < 5; k++) merge(); draw(); };
  el.querySelector(".reset").onclick = () => { parse(); draw(); };
  ta.oninput = () => { parse(); draw(); };
  parse(); draw();
};
