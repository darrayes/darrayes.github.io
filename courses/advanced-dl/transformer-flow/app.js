/* Page logic for the Tiny Transformer visualiser. Model + autograd live in model.js. */
(function () {
'use strict';
const TF = window.TF, $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tex = (el, s) => { if (window.katex) katex.render(s, el, { throwOnError: false, displayMode: false }); else el.textContent = s; };

/* ------------------------------------------------------------------ state */
const S = {
  seed: 2, P: null, st: null, rng: null, step: 0, hist: [], snaps: [], view: 0,
  content: TF.parseContent('abc'), phase: 'fwd', pos: 0, show: 'grad',
  running: false, follow: true, upd: { param: 'dec.cross.Wq', lr: 0.05 }
};
const MAX_STEPS = 600, SNAP_EVERY = 5, LR = 0.01, BATCH = 32;
let RUN = null, lastStageKey = '';

function reset(seed) {
  S.seed = seed; S.P = TF.makeParams(seed); S.st = {}; S.rng = TF.mulberry32(seed * 7919 + 1);
  S.step = 0; S.hist = []; S.snaps = [{ step: 0, v: TF.snapshot(S.P) }]; S.view = 0; S.follow = true;
}
function compute() {
  const P = TF.fromSnapshot(S.P, S.snaps[S.view].v);
  const ex = TF.makeExample(S.content), f = TF.forward(P, ex);
  f.tp.backward(f.loss);
  const uses = new Map();
  for (const n of f.tp.nodes) for (const t of n.ins) uses.set(t, (uses.get(t) || 0) + 1);
  RUN = { P, ex, f, nodes: f.tp.nodes, uses };
}

/* ------------------------------------------------------------------ formatting & colour */
function fmt(x) {
  if (x === -Infinity) return '−∞';
  if (!isFinite(x)) return '∞';
  const a = Math.abs(x);
  if (a < 5e-5) return '0';
  if (a >= 100) return x.toFixed(0);
  if (a >= 10) return x.toFixed(1);
  if (a >= 0.01) return x.toFixed(2).replace(/^(-?)0\./, '$1.');
  return x.toExponential(0);
}
const RGB = { pos: '42,120,214', neg: '226,104,52', gpos: '194,55,138', gneg: '43,154,143' };
function paint(td, x, scale, look, grad) {
  td.className = 'c';
  if (look === 'onehot') { td.classList.add(x ? 'oh1' : 'oh0'); td.textContent = x ? '1' : '0'; return; }
  if (look === 'mask') { td.classList.add(x ? 'mk1' : 'mk0'); td.textContent = x ? '✓' : '−∞'; return; }
  if (x === -Infinity) { td.classList.add('ninf'); td.textContent = '−∞'; return; }
  td.textContent = fmt(x);
  const a = look === 'prob' && !grad ? x : x / (scale || 1);
  const col = grad ? (a >= 0 ? RGB.gpos : RGB.gneg) : (a >= 0 ? RGB.pos : RGB.neg);
  const al = Math.min(1, Math.abs(a)) * 0.88;
  td.style.background = `rgba(${col},${al.toFixed(3)})`;
  td.style.color = al > 0.5 ? '#fff' : '';
}
const rl = (t, i) => t.rows ? t.rows[i] : i;
const cl = (t, j) => t.cols ? t.cols[j] : j;

/* ------------------------------------------------------------------ matrix card */
function card(t, data, o) {
  const el = document.createElement('div');
  el.className = 'card' + (o.role === 'out' ? ' out' : '') + (o.faded ? ' const' : '');
  const kind = o.grad ? '<span class="kind grad">gradient</span>'
    : t.kind === 'param' ? '<span class="kind param">learned</span>'
    : t.kind === 'const' ? '<span class="kind">constant</span>' : '';
  el.innerHTML = `<div class="ch"><b>${esc(o.title)}</b><span class="sh">${t.r}×${t.c}</span>${kind}</div>` + (o.sub ? `<div class="sub">${esc(o.sub)}</div>` : '');
  const tb = document.createElement('table'); tb.className = 'mx';
  let h = '<thead><tr><th></th>';
  for (let j = 0; j < t.c; j++) h += `<th>${esc(cl(t, j))}</th>`;
  h += '</tr></thead><tbody>';
  for (let i = 0; i < t.r; i++) {
    h += `<tr><th title="${esc(rl(t, i))}">${esc(rl(t, i))}</th>`;
    for (let j = 0; j < t.c; j++) h += `<td data-i="${i}" data-j="${j}"></td>`;
    h += '</tr>';
  }
  tb.innerHTML = h + '</tbody>';
  let scale = 0;
  for (const x of data) if (isFinite(x)) scale = Math.max(scale, Math.abs(x));
  const tds = tb.querySelectorAll('td'), cells = [];
  tds.forEach((td, q) => {
    paint(td, data[q], scale, o.look, o.grad);
    if (o.sweep) td.style.animationDelay = Math.min(q * (o.sweepRev ? 1 : 1) * 9, 700) + 'ms', td.classList.add('sw');
    (cells[+td.dataset.i] = cells[+td.dataset.i] || [])[+td.dataset.j] = td;
  });
  if (o.sweep && o.sweepRev) tds.forEach((td, q) => td.style.animationDelay = Math.min((tds.length - 1 - q) * 9, 700) + 'ms');
  el.appendChild(tb);
  el._cells = cells; el._t = t; el._data = data;
  return el;
}

/* ------------------------------------------------------------------ map */
const GROUPS = [
  ['enc', 'Encoder', [['enc.in', 'Embed + position', 'in'], ['enc.attn', 'Self-attention', 'attn', 'padding mask'], ['enc.an1', 'Add & Norm', 'norm'], ['enc.ffn', 'Feed-forward', 'ffn'], ['enc.an2', 'Add & Norm', 'norm', 'output M']]],
  ['dec', 'Decoder', [['dec.in', 'Embed + position', 'in'], ['dec.self', 'Masked self-attention', 'attn', 'causal mask'], ['dec.an1', 'Add & Norm', 'norm'], ['dec.cross', 'Cross-attention', 'cross', 'Q decoder · K,V from M'], ['dec.an2', 'Add & Norm', 'norm'], ['dec.ffn', 'Feed-forward', 'ffn'], ['dec.an3', 'Add & Norm', 'norm'], ['dec.out', 'Output', 'out', 'linear · softmax · loss']]]
];
const GTITLE = {};
GROUPS.forEach(([, side, gs]) => gs.forEach(g => GTITLE[g[0]] = side + ' › ' + g[1]));

function buildMap() {
  let h = '';
  for (const [cls, name, gs] of GROUPS) {
    h += `<div class="col ${cls}"><div class="colh"><span>${name}</span><span>×1 layer</span></div>`;
    for (const [g, title, kind, small] of gs) {
      h += `<div class="gbox ${kind}"><div class="gt"><span>${title}</span>${small ? `<small>${esc(small)}</small>` : ''}</div><div class="chips">`;
      RUN.nodes.forEach((n, i) => { if (n.group === g) h += `<button type="button" class="chip" data-idx="${i}" title="${esc(n.title)}">${esc(n.name)}</button>`; });
      h += '</div></div>';
    }
    if (cls === 'enc') h += '<div class="mlink">M → decoder cross-attention (K, V)</div>';
    h += '</div>';
  }
  $('#map').innerHTML = h;
}
function updateMap() {
  const N = RUN.nodes.length, cur = curIdx();
  document.querySelectorAll('#map .chip').forEach(ch => {
    const i = +ch.dataset.idx;
    ch.className = 'chip' + (S.phase === 'upd' ? ' gdone'
      : i === cur ? ' now'
      : S.phase === 'fwd' ? (i < cur ? ' done' : '')
      : (i > cur ? ' gdone' : ' done'));
  });
  void N;
}

/* ------------------------------------------------------------------ explanations */
const kindOf = n => n.key.split('.').pop();
const isCrossKV = n => n.key === 'dec.cross.k' || n.key === 'dec.cross.v';
function fNote(n) {
  const k = kindOf(n), g = n.group;
  switch (k) {
    case 'emb': return 'Each token selects one row of the embedding matrix E; the one-hot matrix only marks which row. Hover an output cell: it comes from exactly one cell of E. Encoder and decoder share the same E.';
    case 'pe': return g === 'enc.in'
      ? 'Attention has no built-in sense of order, so a fixed sine/cosine pattern is added, one row per position. The source is left-padded and starts with <s>, so the last letter always sits at position 4. That makes "read backwards" easy to learn.'
      : 'The same positional pattern is added in the decoder. The decoder input is the target shifted right by one: it starts with <s>, and each position must predict the token one step ahead.';
    case 'q': return n.group === 'dec.cross'
      ? 'In cross-attention the queries come from the decoder. Each output position asks: which source letter do I need now?'
      : 'Every position\'s row is multiplied by the same learned matrix W_Q. Hover a cell: it depends only on its own row of the input, so no information moves between tokens yet.';
    case 'k': return isCrossKV(n) ? 'Keys and values come from M, the encoder output. This is the only place where the decoder reads the source.' : 'Keys are made the same way, using their own matrix W_K.';
    case 'v': return isCrossKV(n) ? 'Values also come from the encoder output M. They are what will actually be copied into the decoder.' : 'Values carry the content that is passed along once attention has decided where to look.';
    case 's': return 'S[i, j] is the dot product of query row i and key row j, divided by √8. A large value means a good match. Hover a cell to see the two rows it combines.';
    case 'mask': return g === 'enc.attn' ? 'Columns that belong to pad are set to −∞, so no position can attend to padding. The mask is a constant with nothing to learn.'
      : g === 'dec.self' ? 'Position i may only look at positions ≤ i; future cells become −∞. The whole target is fed in at once during training, and this mask stops each position from peeking at the answer.'
      : 'Decoder positions may look at any real source token, but not at padding.';
    case 'attn': return 'Softmax turns each row into weights that sum to 1. Because exp(−∞) = 0, blocked cells get exactly zero weight.' +
      (g === 'dec.cross' ? ' After training this matrix becomes an anti-diagonal: output 1 looks at the last letter, output 2 at the one before it, and so on.' : '');
    case 'av': return 'Each output row is a weighted average of the rows of V. This is where information moves between positions. Hover a cell: it uses one row of A and one whole column of V.';
    case 'o': return 'A final learned matrix maps the attended features back into the model\'s 8-dimensional space.';
    case 'res': return 'The sublayer\'s output is added to its own input, so the original signal also skips around the sublayer unchanged.';
    case 'ln': return 'Each row is shifted to mean 0 and scaled to variance 1, then multiplied by γ and shifted by β. One output cell depends on its whole row.' +
      (n.key === 'enc.ln2.ln' ? ' The result is M, the encoder output: everything the decoder will ever know about the source.' : '');
    case 'l1': return 'The feed-forward network works on each position separately: one row in, one row out. First it expands 8 → 16 features.';
    case 'relu': return 'Negative values are set to zero.';
    case 'l2': return 'Then it projects 16 → 8 features again.';
    case 'logits': return 'One score per vocabulary token, for each decoder position.';
    case 'probs': return 'Softmax again: row t is the model\'s probability distribution for output position t. Before training these are close to uniform (about 1/8 each).';
    case 'loss': return 'Only the probability of the correct token in each row matters: L is the average of −log P over those cells. Rows marked — are padding and ignored. This single number is where the backward pass starts.';
  }
  return '';
}
function bNote(n) {
  const k = kindOf(n);
  switch (k) {
    case 'loss': return 'The backward pass starts with ∂L/∂L = 1. Only the cells of the correct tokens get gradient, −1/(n·P): the smaller the probability, the stronger the push.';
    case 'probs': return 'Softmax spreads that gradient over the whole row: the correct token\'s logit is pushed up (negative gradient) and all the others are pushed down.';
    case 'attn': return 'Softmax backward. Cells with zero attention weight get zero gradient, and that includes every masked cell.';
    case 'mask': return 'The mask passes gradient through where attention was allowed and blocks it (exactly 0) where it was −∞. Masked positions learn nothing from this example.';
    case 's': return 'The score gradient splits into the query side and the key side, each scaled by 1/√8.';
    case 'av': return 'Gradient flows into A (where to look) and into V (what was passed along).';
    case 'res': return 'Addition copies the gradient unchanged into both branches. This is the "gradient highway": the input receives ∂L/∂R directly, however small the gradient through the sublayer is.';
    case 'ln': return 'Layer norm couples each row: one cell\'s gradient depends on its whole row. γ and β get their own gradients, summed over positions.';
    case 'relu': return 'Gradient passes only where U was positive. Everywhere else it is blocked (exactly 0).';
    case 'emb': return 'Only the rows of E for tokens that actually appear receive gradient. Rows of unused tokens stay 0.';
    case 'pe': return 'The positional encoding is a constant: the gradient passes through the addition to the embeddings and stops there.';
    case 'logits': return 'From here the gradient splits: ∂L/∂W_out collects contributions from every position, and ∂L/∂D₃ keeps flowing backwards.';
  }
  if (isCrossKV(n)) return 'For Y = XW the gradient is ∂L/∂X = ∂L/∂Y·Wᵀ and ∂L/∂W = Xᵀ·∂L/∂Y.';
  return 'For Y = XW the gradient is ∂L/∂X = ∂L/∂Y·Wᵀ (it keeps flowing back) and ∂L/∂W = Xᵀ·∂L/∂Y (used to update W). Hover a cell of ∂L/∂W: it collects from a whole column of ∂L/∂Y, one term per position.';
}
const BTEX = {
  linear: '\\frac{\\partial L}{\\partial X}=\\frac{\\partial L}{\\partial Y}W^{\\top}\\qquad \\frac{\\partial L}{\\partial W}=X^{\\top}\\frac{\\partial L}{\\partial Y}\\qquad \\frac{\\partial L}{\\partial b_j}=\\sum_i \\frac{\\partial L}{\\partial Y_{ij}}',
  matmul: '\\frac{\\partial L}{\\partial X}=\\frac{\\partial L}{\\partial Y}W^{\\top}\\qquad \\frac{\\partial L}{\\partial W}=X^{\\top}\\frac{\\partial L}{\\partial Y}',
  scores: '\\frac{\\partial L}{\\partial Q}=\\tfrac{1}{\\sqrt d}\\,\\frac{\\partial L}{\\partial S}K\\qquad \\frac{\\partial L}{\\partial K}=\\tfrac{1}{\\sqrt d}\\Big(\\frac{\\partial L}{\\partial S}\\Big)^{\\!\\top}Q',
  mask: "\\frac{\\partial L}{\\partial S_{ij}}=\\begin{cases}\\partial L/\\partial S'_{ij} & \\text{allowed}\\\\ 0 & \\text{blocked}\\end{cases}",
  softmax: '\\frac{\\partial L}{\\partial X_{ij}}=P_{ij}\\Big(\\frac{\\partial L}{\\partial P_{ij}}-\\sum_k P_{ik}\\frac{\\partial L}{\\partial P_{ik}}\\Big)',
  relu: '\\frac{\\partial L}{\\partial U_{ij}}=\\frac{\\partial L}{\\partial U^+_{ij}}\\cdot[\\,U_{ij}>0\\,]',
  add: '\\frac{\\partial L}{\\partial A}=\\frac{\\partial L}{\\partial B}=\\frac{\\partial L}{\\partial R}\\quad\\text{(copied to both inputs)}',
  layernorm: '\\frac{\\partial L}{\\partial \\gamma_j}=\\sum_i \\frac{\\partial L}{\\partial Y_{ij}}\\hat x_{ij}\\qquad \\frac{\\partial L}{\\partial \\beta_j}=\\sum_i \\frac{\\partial L}{\\partial Y_{ij}}\\qquad \\frac{\\partial L}{\\partial X_{i}}=\\tfrac{1}{\\sigma_i}\\big(g_i-\\bar g_i-\\hat x_i\\,\\overline{g\\hat x}_i\\big)',
  embed: '\\frac{\\partial L}{\\partial E_{\\text{tok}}} = \\sum_{i\\,:\\,\\text{token}_i=\\text{tok}} \\frac{\\partial L}{\\partial X_{\\text{emb},i}}',
  nll: '\\frac{\\partial L}{\\partial P_{t,y_t}}=-\\frac{1}{n\\,P_{t,y_t}}\\qquad\\text{(0 everywhere else)}'
};

/* ------------------------------------------------------------------ stage view */
const N = () => RUN.nodes.length;
const curIdx = () => S.phase === 'bwd' ? N() - 1 - S.pos : S.pos;
let CARDS = [];   // [{el, role:'in'|'out', k}]

function stageShell() {
  const st = $('#stage');
  st.innerHTML = `
    <div class="stagebar">
      <button type="button" id="prev" aria-label="Previous step">◀ Prev</button>
      <button type="button" id="play">▶ Play</button>
      <button type="button" id="next" aria-label="Next step">Next ▶</button>
      <input type="range" id="pos" min="0" value="0" aria-label="Step">
      <span class="cnt" id="cnt"></span>
    </div>
    <div class="path" id="path"></div>
    <div class="stitle" id="stitle"></div>
    <div class="formula" id="formula"></div>
    <p class="note" id="note"></p>
    <div id="extra"></div>
    <div id="bshow" class="legend" hidden><label><input type="checkbox" id="showvals"> show forward values instead of gradients</label></div>
    <div class="mats" id="mats"></div>
    <div class="readout" id="readout">Hover a cell.</div>
    <div class="legend" id="legend"></div>`;
  $('#prev').onclick = () => go(S.pos - 1);
  $('#next').onclick = () => go(S.pos + 1);
  $('#pos').oninput = e => go(+e.target.value);
  $('#play').onclick = togglePlay;
  $('#showvals').onchange = e => { S.show = e.target.checked ? 'val' : 'grad'; renderStage(true); };
  const mats = $('#mats');
  mats.addEventListener('mouseover', onHover);
  mats.addEventListener('mouseleave', clearHover);
}
let playT = null;
function togglePlay() {
  if (playT) { clearInterval(playT); playT = null; $('#play').textContent = '▶ Play'; return; }
  if (S.pos >= N() - 1) go(0);
  $('#play').textContent = '❚❚ Pause';
  playT = setInterval(() => { if (S.pos >= N() - 1) togglePlay(); else go(S.pos + 1); }, 1600);
}
function go(p) {
  if (S.phase === 'upd') return;
  S.pos = Math.max(0, Math.min(N() - 1, p));
  renderStage();
}

function renderStage(force) {
  const wrap = $('#stagewrap');
  wrap.classList.toggle('bwd', S.phase === 'bwd');
  updateMap();
  if (S.phase === 'upd') return renderUpdate(force);
  ['prev', 'play', 'next', 'pos'].forEach(id => $('#' + id).disabled = false);
  const n = RUN.nodes[curIdx()], bwd = S.phase === 'bwd';
  const key = S.phase + ':' + n.key + ':' + S.show;
  const sweep = key !== lastStageKey; lastStageKey = key;
  $('#pos').max = N() - 1; $('#pos').value = S.pos;
  $('#cnt').textContent = `${S.pos + 1} / ${N()}`;
  $('#path').textContent = (bwd ? 'Backward · ' : 'Forward · ') + GTITLE[n.group];
  $('#stitle').textContent = n.title;
  tex($('#formula'), bwd ? BTEX[n.type] : n.tex);
  $('#note').textContent = bwd ? bNote(n) : fNote(n);
  // extra notes (backward): branching + the encoder's only gradient route
  let ex = '';
  if (bwd) {
    if (isCrossKV(n)) ex += '<p class="note extra star">★ This is the <b>only route</b> by which gradient reaches the encoder: through the keys and values of cross-attention. Everything the encoder learns, it learns through here.</p>';
    n.ins.forEach((t, k) => {
      const u = RUN.uses.get(t);
      if (n.contrib[k] && u > 1) ex += `<p class="note extra">${esc(t.name)} feeds ${u} operations, so its full gradient is the <b>sum of ${u} contributions</b> like the one shown here.</p>`;
    });
  }
  $('#extra').innerHTML = ex;
  $('#bshow').hidden = !bwd;
  // cards
  const mats = $('#mats'); mats.innerHTML = ''; mats.classList.remove('hovering'); CARDS = [];
  const look = t => t.look || (t.node && t.node.type === 'softmax' ? 'prob' : null);
  const showGrad = bwd && S.show === 'grad';
  const outCard = () => {
    const t = n.out, el = card(t, showGrad ? n.outGrad : t.v, {
      title: showGrad ? '∂L/∂' + t.name : t.name, role: 'out', grad: showGrad, look: showGrad ? null : look(t),
      sub: n.type === 'nll' ? 'L = ' + t.v[0].toFixed(4) : '', sweep: sweep && !bwd
    });
    CARDS.push({ el, role: 'out' }); return el;
  };
  const inCards = () => n.ins.map((t, k) => {
    const g = showGrad && n.contrib[k];
    const share = g && RUN.uses.get(t) > 1 ? `this operation's share (1 of ${RUN.uses.get(t)} contributions)` : '';
    const el = card(t, g ? n.contrib[k] : t.v, {
      title: g ? '∂L/∂' + t.name : t.name, role: 'in', grad: !!g, look: g ? null : look(t),
      faded: showGrad && !g, sub: showGrad && !g ? 'constant: no gradient' : share, sweep: sweep && bwd && !!g, sweepRev: true
    });
    CARDS.push({ el, role: 'in', k }); return el;
  });
  const glyph = s => { const d = document.createElement('div'); d.className = 'glyph'; d.textContent = s; return d; };
  if (!bwd) { inCards().forEach(c => mats.appendChild(c)); mats.appendChild(glyph('→')); mats.appendChild(outCard()); }
  else { mats.appendChild(outCard()); mats.appendChild(glyph('→')); inCards().forEach(c => mats.appendChild(c)); }
  $('#readout').innerHTML = bwd
    ? 'Hover a cell of <b>∂L/∂' + esc(n.out.name) + '</b> to see where its gradient goes, or a gradient cell on the right to see where it came from.'
    : 'Hover a cell of <b>' + esc(n.out.name) + '</b> to see what it was computed from, or an input cell to see what it feeds.';
  $('#legend').innerHTML = showGrad
    ? `<span><i style="background:linear-gradient(90deg,rgb(${RGB.gneg}),#f4f3ef,rgb(${RGB.gpos}))"></i>gradient: − 0 +</span><span><i style="background:repeating-linear-gradient(45deg,#e6e4dd 0 3px,#f5f4f0 3px 6px)"></i>masked</span>`
    : `<span><i style="background:linear-gradient(90deg,rgb(${RGB.neg}),#f4f3ef,rgb(${RGB.pos}))"></i>value: − 0 + (scaled per matrix)</span><span><i style="background:repeating-linear-gradient(45deg,#e6e4dd 0 3px,#f5f4f0 3px 6px)"></i>masked (−∞)</span>`;
}

/* hover tracing */
function clearHover() {
  $('#mats').classList.remove('hovering');
  document.querySelectorAll('#mats td.hl, #mats td.me').forEach(td => td.classList.remove('hl', 'me'));
}
function depText(t, cells) {
  const nm = t.name, n = cells.length;
  if (!n) return '';
  if (n === 1) return `${nm}[${rl(t, cells[0][0])}, ${cl(t, cells[0][1])}]`;
  const rows = new Set(cells.map(x => x[0])), cols = new Set(cells.map(x => x[1]));
  if (rows.size === 1 && n === t.c) return `row “${rl(t, cells[0][0])}” of ${nm}`;
  if (cols.size === 1 && n === t.r) return `column ${cl(t, cells[0][1])} of ${nm}`;
  return `${n} cells of ${nm}`;
}
function onHover(e) {
  const td = e.target.closest('td[data-i]'); if (!td) return;
  const cardEl = td.closest('.card'), C = CARDS.find(c => c.el === cardEl); if (!C) return;
  clearHover();
  const i = +td.dataset.i, j = +td.dataset.j;
  if (S.phase === 'upd') return hoverUpdate(i, j, C);
  const n = RUN.nodes[curIdx()], bwd = S.phase === 'bwd', showGrad = bwd && S.show === 'grad';
  const outC = CARDS.find(c => c.role === 'out'), inC = k => CARDS.find(c => c.role === 'in' && c.k === k);
  const mark = (c, cells) => cells.forEach(([r, q]) => { const x = c.el._cells[r] && c.el._cells[r][q]; if (x) x.classList.add('hl'); });
  $('#mats').classList.add('hovering');
  td.classList.add('hl', 'me');
  const val = x => fmt(x);
  if (C.role === 'out') {
    const deps = n.deps(i, j);
    deps.forEach(d => mark(inC(d.t), d.cells));
    const what = deps.map(d => depText(n.ins[d.t], d.cells)).filter(Boolean).join(' and ');
    const o = n.out, nm = showGrad ? '∂L/∂' + o.name : o.name;
    $('#readout').innerHTML = `<b>${esc(nm)}[${esc(rl(o, i))}, ${esc(cl(o, j))}] = ${val(cardEl._data[i * o.c + j])}</b><br>` +
      (bwd ? 'This gradient flows back into: ' : 'Computed from: ') + esc(what) + '.' +
      (n.type === 'mask' && !n.ins[1].v[i * o.c + j] ? ' This position is <b>blocked</b>.' : '');
  } else {
    const t = n.ins[C.k], hits = [];
    for (let a = 0; a < n.out.r; a++) for (let b = 0; b < n.out.c; b++)
      if (n.deps(a, b).some(d => d.t === C.k && d.cells.some(([r, q]) => r === i && q === j))) hits.push([a, b]);
    mark(outC, hits);
    const nm = (showGrad && n.contrib[C.k] ? '∂L/∂' : '') + t.name;
    $('#readout').innerHTML = `<b>${esc(nm)}[${esc(rl(t, i))}, ${esc(cl(t, j))}] = ${val(cardEl._data[i * t.c + j])}</b><br>` +
      (hits.length ? (bwd && showGrad ? 'Collects gradient from ' : 'Feeds ') + esc(depText(n.out, hits)) + ' of ' + (showGrad ? '∂L/∂' : '') + esc(n.out.name) + '.'
        : 'Feeds nothing in this operation.');
  }
}

/* ------------------------------------------------------------------ update phase */
const PARAM_LABEL = k => k.replace('enc.', 'Encoder ').replace('dec.', 'Decoder ').replace('self.', 'self-attn ').replace('cross.', 'cross-attn ')
  .replace('ffn.', 'FFN ').replace('ln', 'LayerNorm ').replace('out.', 'Output ').replace(/\.g$/, ' γ').replace(/\.b$/, ' β');
function totalGrad(pname) {
  let g = null;
  for (const l of RUN.f.tp.leaves) if (l.kind === 'param' && l.pname === pname && l.g) {
    if (!g) g = new Float64Array(l.g.length);
    for (let q = 0; q < g.length; q++) g[q] += l.g[q];
  }
  return g || new Float64Array(RUN.P[pname].v.length);
}
function renderUpdate() {
  ['prev', 'play', 'next', 'pos'].forEach(id => $('#' + id).disabled = true);
  $('#cnt').textContent = '';
  $('#path').textContent = 'Update · one gradient-descent step';
  $('#stitle').textContent = 'Every weight moves a little against its gradient';
  tex($('#formula'), 'W \\leftarrow W - \\eta\\,\\frac{\\partial L}{\\partial W}');
  $('#note').textContent = 'The backward pass produced a gradient for every learned matrix. Pick one to see the update cell by cell. A positive gradient means "increasing this weight increases the loss", so the weight goes down.';
  $('#bshow').hidden = true;
  const P = RUN.P, names = Object.keys(P);
  if (!$('#updsel')) {
    $('#extra').innerHTML = `<div class="updctl"><label>Parameter <select id="updsel">${names.map(k => `<option value="${k}">${esc(PARAM_LABEL(k))} (${P[k].r}×${P[k].c})</option>`).join('')}</select></label>
      <label>Step size η <input type="range" id="updlr" min="0.005" max="1" step="0.005"> <b id="updlrv"></b></label></div><div class="lossbox" id="lossbox"></div>`;
    $('#updsel').onchange = e => { S.upd.param = e.target.value; renderUpdate(); };
    $('#updlr').oninput = e => { S.upd.lr = +e.target.value; renderUpdate(); };
  }
  $('#updsel').value = S.upd.param; $('#updlr').value = S.upd.lr; $('#updlrv').textContent = S.upd.lr.toFixed(3);
  const pn = S.upd.param, p = P[pn], g = totalGrad(pn), nv = p.v.map((x, q) => x - S.upd.lr * g[q]);
  const shape = { r: p.r, c: p.c, rows: pn === 'E' ? TF.VOCAB : null, cols: pn.startsWith('out.') ? TF.VOCAB : null, kind: 'param' };
  const mats = $('#mats'); mats.innerHTML = ''; mats.classList.remove('hovering'); CARDS = [];
  const add = (data, o) => { const el = card(Object.assign({ name: o.title }, shape), data, o); CARDS.push({ el, role: o.role }); mats.appendChild(el); };
  const glyph = s => { const d = document.createElement('div'); d.className = 'glyph'; d.textContent = s; mats.appendChild(d); };
  add(p.v, { title: 'W (before)', role: 'a' }); glyph('−');
  add(g, { title: '∂L/∂W', role: 'b', grad: true, sub: 'total for this example' }); glyph('=');
  add(nv, { title: 'W (after)', role: 'c' });
  // loss before/after when ALL parameters take this step
  const Q = TF.fromSnapshot(P, TF.snapshot(P));
  for (const k of names) { const gk = totalGrad(k); for (let q = 0; q < gk.length; q++) Q[k].v[q] -= S.upd.lr * gk[q]; }
  const after = TF.forward(Q, RUN.ex).loss.v[0], before = RUN.f.loss.v[0];
  $('#lossbox').innerHTML = `<div class="stat">Loss on this example now<b>${before.toFixed(3)}</b></div>
    <div class="stat">After this step on <em>all</em> ${names.length} parameter tensors<b style="color:${after < before ? 'var(--v)' : '#d6492a'}">${after.toFixed(3)}</b></div>
    <div class="stat" style="max-width:300px;align-self:center">${after < before ? 'One step already helps on this example.' : 'Too large a step overshoots, so the loss goes up. Try a smaller η.'} Training (section 3) repeats this hundreds of times, each time on a fresh batch of examples.</div>`;
  $('#readout').innerHTML = 'Hover a cell: the same position in all three matrices lights up.';
  $('#legend').innerHTML = '';
}
function hoverUpdate(i, j) {
  $('#mats').classList.add('hovering');
  CARDS.forEach(c => { const x = c.el._cells[i] && c.el._cells[i][j]; if (x) x.classList.add('hl'); });
  const [a, b, c] = CARDS.map(C => C.el._data[i * C.el._t.c + j]);
  const f4 = x => x.toFixed(4);
  $('#readout').innerHTML = `<b>${f4(c)}</b> = ${f4(a)} − ${S.upd.lr.toFixed(3)} × (${f4(b)})`;
}

/* ------------------------------------------------------------------ task panel */
function tokHtml(t, i, ign) {
  const name = TF.VOCAB[t], cls = ign ? 'ign' : t === TF.PAD ? 'pad' : t <= 2 ? 'special' : '';
  return `<span class="tok ${cls}">${esc(ign ? '—' : name)}<small>${i}</small></span>`;
}
function renderTask() {
  const ex = TF.makeExample(S.content);
  $('#seqs').innerHTML =
    `<span class="lbl">Encoder input</span><span class="toks">${ex.src.map((t, i) => tokHtml(t, i)).join('')}</span>
     <span class="lbl">Decoder input</span><span class="toks">${ex.decIn.map((t, i) => tokHtml(t, i)).join('')}</span>
     <span class="lbl">Target output</span><span class="toks">${ex.tgt.map((t, i) => tokHtml(t, i, !ex.w[i])).join('')}</span>`;
}
function setContent(c) {
  S.content = c; renderTask(); compute(); renderStage(true); renderTrainViews();
}

/* ------------------------------------------------------------------ training panel */
function trainSteps(k) {
  for (let s = 0; s < k && S.step < MAX_STEPS; s++) {
    const o = TF.trainStep(S.P, S.st, S.rng, BATCH, LR);
    S.step++; S.hist.push([S.step, o.loss, o.acc]);
    if (S.step % SNAP_EVERY === 0) S.snaps.push({ step: S.step, v: TF.snapshot(S.P) });
  }
  if (S.follow) S.view = S.snaps.length - 1;
}
let frame = 0, acc = 0;
function loop() {
  if (!S.running) return;
  acc += +$('#speed').value;
  const k = Math.floor(acc); acc -= k;
  if (k) trainSteps(k);
  renderChart();
  if (++frame % 6 === 0) { compute(); renderStage(); renderTrainViews(); }
  if (S.step >= MAX_STEPS) stopTraining();
  else requestAnimationFrame(loop);
}
function stopTraining() {
  S.running = false; $('#trainbtn').textContent = S.step >= MAX_STEPS ? 'Done' : '▶ Train';
  $('#trainbtn').disabled = S.step >= MAX_STEPS;
  compute(); renderStage(); renderTrainViews(); renderChart();
}
function renderChart() {
  const W = 520, H = 220, l = 38, r = 10, t = 10, b = 28, ymax = 3;
  const xmax = Math.max(300, Math.ceil(S.step / 100) * 100);
  const X = s => l + (W - l - r) * s / xmax, Y = v => t + (H - t - b) * (1 - Math.min(v, ymax) / ymax);
  let h = '';
  for (let v = 0; v <= ymax; v += 0.5) h += `<line x1="${l}" x2="${W - r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#eceae4"/><text x="${l - 6}" y="${Y(v) + 3.5}" font-size="10" text-anchor="end" fill="#85837d">${v}</text>`;
  for (let s = 0; s <= xmax; s += xmax / 5) h += `<text x="${X(s)}" y="${H - 10}" font-size="10" text-anchor="middle" fill="#85837d">${s}</text>`;
  h += `<text x="${W - r}" y="${H - 10 - 12}" font-size="10" text-anchor="end" fill="#85837d">training step</text>`;
  if (S.hist.length) {
    // light raw curve + smoothed curve
    const pts = S.hist.map(([s, L]) => `${X(s).toFixed(1)},${Y(L).toFixed(1)}`).join(' ');
    let sm = 0, first = true;
    const spts = S.hist.map(([s, L]) => { sm = first ? L : 0.9 * sm + 0.1 * L; first = false; return `${X(s).toFixed(1)},${Y(sm).toFixed(1)}`; }).join(' ');
    h += `<polyline points="${pts}" fill="none" stroke="#2a78d6" stroke-opacity=".25" stroke-width="1"/><polyline points="${spts}" fill="none" stroke="#2a78d6" stroke-width="2"/>`;
  } else h += `<text x="${(W + l) / 2}" y="${H / 2}" font-size="13" text-anchor="middle" fill="#85837d">Press ▶ Train</text>`;
  const vs = S.snaps[S.view].step;
  h += `<line x1="${X(vs)}" x2="${X(vs)}" y1="${t}" y2="${H - b}" stroke="#1b1b1a" stroke-dasharray="3 3"/>`;
  $('#chart').innerHTML = h;
  const last = S.hist[S.hist.length - 1];
  $('#sStep').textContent = S.step;
  $('#sLoss').textContent = last ? last[1].toFixed(3) : '–';
  $('#sAcc').textContent = last ? Math.round(last[2] * 100) + '%' : '–';
  const sc = $('#scrub'); sc.max = S.snaps.length - 1; sc.value = S.view;
  $('#scrubLbl').textContent = `Viewing weights at step ${vs}` + (S.view === S.snaps.length - 1 ? ' (latest)' : '');
}
function renderTrainViews() {
  const ex = RUN.ex, gen = TF.generate(RUN.P, S.content);
  const want = ex.tgt.slice(0, S.content.length + 1);
  const ok = gen.join() === want.join();
  $('#pred').innerHTML = `<span>Model's own output for <b>${esc(S.content.map(t => TF.VOCAB[t]).join(' '))}</b>:</span>
    <span class="toks">${gen.map((t, i) => tokHtml(t, i)).join('')}</span><span class="${ok ? 'ok' : 'no'}">${ok ? '✓ correct' : '✗ not yet'}</span>`;
  const heat = $('#heat'); heat.innerHTML = '';
  [['enc', 'Encoder self-attention'], ['self', 'Decoder masked self-attention'], ['cross', 'Cross-attention (decoder → source)']].forEach(([k, title]) => {
    heat.appendChild(card(RUN.f.attn[k], RUN.f.attn[k].v, { title, look: 'prob' }));
  });
}

/* ------------------------------------------------------------------ wiring */
function init() {
  reset(S.seed); compute();
  stageShell(); buildMap(); renderTask(); renderStage(); renderChart(); renderTrainViews();

  $('#map').addEventListener('click', e => {
    const ch = e.target.closest('.chip'); if (!ch) return;
    if (S.phase === 'upd') setPhase('fwd');
    const i = +ch.dataset.idx; go(S.phase === 'bwd' ? N() - 1 - i : i);
  });
  $('#phases').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setPhase(b.dataset.ph); });
  function setPhase(ph) {
    if (playT) togglePlay();
    S.phase = ph; S.pos = 0; lastStageKey = '';
    document.querySelectorAll('#phases button').forEach(b => b.classList.toggle('on', b.dataset.ph === ph));
    if (ph !== 'upd') $('#extra').innerHTML = '';
    renderStage(true);
  }
  const inp = $('#inp');
  inp.addEventListener('input', () => {
    const c = TF.parseContent(inp.value);
    inp.classList.toggle('bad', !c);
    $('#inhint').textContent = c ? 'Everything below updates when you change this.' : 'Use 1 to 4 letters from a–e.';
    if (c) setContent(c);
  });
  $('#presets').innerHTML = ['a', 'ab', 'abc', 'abcd', 'cab', 'eed'].map(s => `<button type="button">${s}</button>`).join('');
  $('#presets').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; inp.value = b.textContent; inp.dispatchEvent(new Event('input')); });
  const rr = TF.mulberry32(Date.now() % 100000);
  $('#rand').onclick = () => { inp.value = TF.randomExample(rr).content.map(t => TF.VOCAB[t]).join(''); inp.dispatchEvent(new Event('input')); };

  $('#trainbtn').onclick = () => {
    if (S.running) return stopTraining();
    S.running = true; S.follow = true; $('#trainbtn').textContent = '❚❚ Pause'; loop();
  };
  $('#step1').onclick = () => { if (S.running) return; trainSteps(1); S.follow = true; S.view = S.snaps.length - 1;
    if (S.step % SNAP_EVERY) { S.snaps.push({ step: S.step, v: TF.snapshot(S.P) }); S.view = S.snaps.length - 1; }
    compute(); renderStage(); renderTrainViews(); renderChart(); };
  $('#resetbtn').onclick = () => {
    if (S.running) S.running = false;
    reset(S.seed + 1); $('#trainbtn').disabled = false; $('#trainbtn').textContent = '▶ Train';
    compute(); renderStage(true); renderTrainViews(); renderChart();
  };
  $('#scrub').oninput = e => {
    S.view = +e.target.value; S.follow = S.view === S.snaps.length - 1;
    compute(); renderStage(); renderTrainViews(); renderChart();
  };
  document.addEventListener('keydown', e => {
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    if (e.key === 'ArrowRight') go(S.pos + 1);
    else if (e.key === 'ArrowLeft') go(S.pos - 1);
  });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
