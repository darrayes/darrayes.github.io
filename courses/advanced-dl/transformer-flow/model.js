/* =====================================================================
   A tiny encoder–decoder Transformer with a hand-written autograd.

   Every operation is recorded on a "tape" together with:
     - its inputs and output (so the page can draw them),
     - a backward function (so gradients can flow back),
     - a deps(i, j) function listing the input cells that output cell
       (i, j) was computed from (so the page can trace information flow).

   Sizes are deliberately tiny so every number fits on screen:
     vocabulary 8, d_model 8, d_ff 16, one head, one layer each side,
     source and target length 5.
   ===================================================================== */
(function (root) {
'use strict';

const VOCAB = ['pad', '<s>', '</s>', 'a', 'b', 'c', 'd', 'e'];
const PAD = 0, BOS = 1, EOS = 2;
const LETTERS = ['a', 'b', 'c', 'd', 'e'];
const MAXN = 4;            // at most 4 content tokens
const LS = MAXN + 1;       // source: left padding + <s> + content
const LT = MAXN + 1;       // target: content reversed + </s>, right padded
const D = 8, F = 16, V = VOCAB.length;

/* ---------- random numbers (seeded, so "Reset" is reproducible) ---------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gauss(rng) {
  let u = 0; while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

/* ---------- tensors ---------- */
function T(r, c, v) { this.r = r; this.c = c; this.v = v || new Float64Array(r * c); this.g = null; }

const rowCells = (t, i) => Array.from({ length: t.c }, (_, j) => [i, j]);
const colCells = (t, j) => Array.from({ length: t.r }, (_, i) => [i, j]);

/* ---------- parameters ---------- */
function makeParams(seed) {
  const rng = mulberry32(seed);
  const P = {};
  const p = (name, r, c, init) => {
    const v = new Float64Array(r * c);
    for (let k = 0; k < v.length; k++) v[k] = init === 'one' ? 1 : init === 'zero' ? 0 : gauss(rng) * init;
    P[name] = { name, r, c, v, g: new Float64Array(r * c), m: new Float64Array(r * c), s: new Float64Array(r * c) };
  };
  const w = 1 / Math.sqrt(D), wf = 1 / Math.sqrt(F);
  p('E', V, D, 1);
  for (const pre of ['enc.self', 'dec.self', 'dec.cross'])
    for (const n of ['Wq', 'Wk', 'Wv', 'Wo']) p(pre + '.' + n, D, D, w);
  for (const pre of ['enc.ffn', 'dec.ffn']) {
    p(pre + '.W1', D, F, w); p(pre + '.b1', 1, F, 'zero');
    p(pre + '.W2', F, D, wf); p(pre + '.b2', 1, D, 'zero');
  }
  for (const n of ['enc.ln1', 'enc.ln2', 'dec.ln1', 'dec.ln2', 'dec.ln3']) { p(n + '.g', 1, D, 'one'); p(n + '.b', 1, D, 'zero'); }
  p('out.W', D, V, w); p('out.b', 1, V, 'zero');
  return P;
}
function snapshot(P) { const s = {}; for (const k in P) s[k] = P[k].v.slice(); return s; }
function fromSnapshot(P, snap) {
  const Q = {};
  for (const k in P) Q[k] = { name: k, r: P[k].r, c: P[k].c, v: snap[k], g: new Float64Array(P[k].v.length) };
  return Q;
}

/* ---------- the tape ---------- */
function Tape() { this.nodes = []; this.leaves = []; }
Tape.prototype.leaf = function (r, c, v, info) {
  const t = new T(r, c, v); Object.assign(t, info); this.leaves.push(t); return t;
};
Tape.prototype.konst = function (r, c, v, info) { return this.leaf(r, c, v, Object.assign({ kind: 'const' }, info)); };
Tape.prototype.param = function (P, pname, info) {
  const p = P[pname];
  return this.leaf(p.r, p.c, p.v, Object.assign({ kind: 'param', pname }, info));
};
Tape.prototype.op = function (spec) {
  spec.idx = this.nodes.length; spec.out.node = spec; spec.out.kind = 'act'; spec.out.name = spec.name;
  this.nodes.push(spec); return spec.out;
};
/* Reverse-mode sweep. node.outGrad is dL/d(output) (complete when the node
   is reached); node.contrib[k] is this op's contribution to dL/d(input k). */
Tape.prototype.backward = function (loss, P) {
  for (const n of this.nodes) n.out.g = null;
  for (const l of this.leaves) l.g = null;
  loss.g = new Float64Array([1]);
  for (let n = this.nodes.length - 1; n >= 0; n--) {
    const node = this.nodes[n], out = node.out;
    if (!out.g) out.g = new Float64Array(out.v.length);
    node.outGrad = out.g;
    node.contrib = node.back(out.g);
    node.contrib.forEach((cg, k) => {
      const inp = node.ins[k];
      if (!cg || inp.kind === 'const') { node.contrib[k] = null; return; }
      if (!inp.g) inp.g = new Float64Array(inp.v.length);
      for (let q = 0; q < cg.length; q++) inp.g[q] += cg[q];
    });
  }
  if (P) for (const l of this.leaves) if (l.kind === 'param' && l.g) {
    const g = P[l.pname].g; for (let q = 0; q < g.length; q++) g[q] += l.g[q];
  }
};

/* ---------- operations ---------- */
// Y = X W (+ b)
function linear(tp, X, W, b, meta) {
  const r = X.r, k = X.c, c = W.c, Y = new T(r, c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
    let s = b ? b.v[j] : 0;
    for (let t = 0; t < k; t++) s += X.v[i * k + t] * W.v[t * c + j];
    Y.v[i * c + j] = s;
  }
  Y.rows = X.rows; Y.cols = meta.cols || W.cols;
  return tp.op(Object.assign({
    type: b ? 'linear' : 'matmul', ins: b ? [X, W, b] : [X, W], out: Y,
    back(gY) {
      const gX = new Float64Array(r * k), gW = new Float64Array(k * c), gb = b ? new Float64Array(c) : null;
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const g = gY[i * c + j]; if (!g) continue;
        for (let t = 0; t < k; t++) { gX[i * k + t] += g * W.v[t * c + j]; gW[t * c + j] += g * X.v[i * k + t]; }
        if (gb) gb[j] += g;
      }
      return b ? [gX, gW, gb] : [gX, gW];
    },
    deps(i, j) {
      const d = [{ t: 0, cells: rowCells(X, i) }, { t: 1, cells: colCells(W, j) }];
      if (b) d.push({ t: 2, cells: [[0, j]] });
      return d;
    }
  }, meta));
}
// S = Q K^T / sqrt(d)
function scores(tp, Q, K, meta) {
  const r = Q.r, c = K.r, k = Q.c, sc = 1 / Math.sqrt(k), S = new T(r, c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
    let s = 0; for (let t = 0; t < k; t++) s += Q.v[i * k + t] * K.v[j * k + t];
    S.v[i * c + j] = s * sc;
  }
  S.rows = Q.rows; S.cols = K.rows;
  return tp.op(Object.assign({
    type: 'scores', ins: [Q, K], out: S,
    back(gS) {
      const gQ = new Float64Array(r * k), gK = new Float64Array(c * k);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
        const g = gS[i * c + j] * sc; if (!g) continue;
        for (let t = 0; t < k; t++) { gQ[i * k + t] += g * K.v[j * k + t]; gK[j * k + t] += g * Q.v[i * k + t]; }
      }
      return [gQ, gK];
    },
    deps(i, j) { return [{ t: 0, cells: rowCells(Q, i) }, { t: 1, cells: rowCells(K, j) }]; }
  }, meta));
}
// blocked positions become -infinity
function mask(tp, S, M, meta) {
  const Y = new T(S.r, S.c);
  for (let q = 0; q < Y.v.length; q++) Y.v[q] = M.v[q] ? S.v[q] : -Infinity;
  Y.rows = S.rows; Y.cols = S.cols;
  return tp.op(Object.assign({
    type: 'mask', ins: [S, M], out: Y,
    back(g) { const gS = new Float64Array(g.length); for (let q = 0; q < g.length; q++) gS[q] = M.v[q] ? g[q] : 0; return [gS, null]; },
    deps(i, j) { return [{ t: 0, cells: [[i, j]] }, { t: 1, cells: [[i, j]] }]; }
  }, meta));
}
function softmax(tp, X, meta) {
  const r = X.r, c = X.c, Y = new T(r, c);
  for (let i = 0; i < r; i++) {
    let m = -Infinity; for (let j = 0; j < c; j++) m = Math.max(m, X.v[i * c + j]);
    let s = 0; for (let j = 0; j < c; j++) { const e = Math.exp(X.v[i * c + j] - m); Y.v[i * c + j] = e; s += e; }
    for (let j = 0; j < c; j++) Y.v[i * c + j] /= s;
  }
  Y.rows = X.rows; Y.cols = X.cols;
  return tp.op(Object.assign({
    type: 'softmax', ins: [X], out: Y,
    back(gY) {
      const gX = new Float64Array(r * c);
      for (let i = 0; i < r; i++) {
        let dot = 0; for (let j = 0; j < c; j++) dot += gY[i * c + j] * Y.v[i * c + j];
        for (let j = 0; j < c; j++) gX[i * c + j] = Y.v[i * c + j] * (gY[i * c + j] - dot);
      }
      return [gX];
    },
    deps(i) { return [{ t: 0, cells: rowCells(X, i) }]; }
  }, meta));
}
function relu(tp, X, meta) {
  const Y = new T(X.r, X.c);
  for (let q = 0; q < Y.v.length; q++) Y.v[q] = X.v[q] > 0 ? X.v[q] : 0;
  Y.rows = X.rows; Y.cols = X.cols;
  return tp.op(Object.assign({
    type: 'relu', ins: [X], out: Y,
    back(g) { const gX = new Float64Array(g.length); for (let q = 0; q < g.length; q++) gX[q] = X.v[q] > 0 ? g[q] : 0; return [gX]; },
    deps(i, j) { return [{ t: 0, cells: [[i, j]] }]; }
  }, meta));
}
function add(tp, A, B, meta) {
  const Y = new T(A.r, A.c);
  for (let q = 0; q < Y.v.length; q++) Y.v[q] = A.v[q] + B.v[q];
  Y.rows = A.rows; Y.cols = A.cols;
  return tp.op(Object.assign({
    type: 'add', ins: [A, B], out: Y,
    back(g) { return [g.slice(), g.slice()]; },
    deps(i, j) { return [{ t: 0, cells: [[i, j]] }, { t: 1, cells: [[i, j]] }]; }
  }, meta));
}
function layernorm(tp, X, G, B, meta) {
  const r = X.r, c = X.c, Y = new T(r, c), xh = new Float64Array(r * c), rstd = new Float64Array(r);
  for (let i = 0; i < r; i++) {
    let mu = 0; for (let j = 0; j < c; j++) mu += X.v[i * c + j]; mu /= c;
    let va = 0; for (let j = 0; j < c; j++) va += (X.v[i * c + j] - mu) ** 2; va /= c;
    rstd[i] = 1 / Math.sqrt(va + 1e-5);
    for (let j = 0; j < c; j++) { xh[i * c + j] = (X.v[i * c + j] - mu) * rstd[i]; Y.v[i * c + j] = xh[i * c + j] * G.v[j] + B.v[j]; }
  }
  Y.rows = X.rows; Y.cols = X.cols;
  return tp.op(Object.assign({
    type: 'layernorm', ins: [X, G, B], out: Y,
    back(gY) {
      const gX = new Float64Array(r * c), gG = new Float64Array(c), gB = new Float64Array(c);
      for (let i = 0; i < r; i++) {
        let m1 = 0, m2 = 0; const dx = new Float64Array(c);
        for (let j = 0; j < c; j++) {
          const g = gY[i * c + j]; gG[j] += g * xh[i * c + j]; gB[j] += g;
          dx[j] = g * G.v[j]; m1 += dx[j]; m2 += dx[j] * xh[i * c + j];
        }
        m1 /= c; m2 /= c;
        for (let j = 0; j < c; j++) gX[i * c + j] = rstd[i] * (dx[j] - m1 - xh[i * c + j] * m2);
      }
      return [gX, gG, gB];
    },
    deps(i, j) { return [{ t: 0, cells: rowCells(X, i) }, { t: 1, cells: [[0, j]] }, { t: 2, cells: [[0, j]] }]; }
  }, meta));
}
// row i of the output is row ids[i] of E  (= one-hot(ids) · E)
function embed(tp, OH, E, ids, meta) {
  const r = ids.length, c = E.c, Y = new T(r, c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) Y.v[i * c + j] = E.v[ids[i] * c + j];
  Y.rows = OH.rows;
  return tp.op(Object.assign({
    type: 'embed', ins: [OH, E], out: Y,
    back(gY) {
      const gE = new Float64Array(E.r * c);
      for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) gE[ids[i] * c + j] += gY[i * c + j];
      return [null, gE];
    },
    deps(i, j) { return [{ t: 0, cells: [[i, ids[i]]] }, { t: 1, cells: [[ids[i], j]] }]; }
  }, meta));
}
// mean negative log-likelihood of the target tokens (rows with no target are ignored)
function nll(tp, Pr, Tg, meta) {
  const Y = new T(1, 1); let s = 0, n = 0;
  for (let q = 0; q < Pr.v.length; q++) if (Tg.v[q]) { s -= Math.log(Math.max(Pr.v[q], 1e-12)); n++; }
  Y.v[0] = n ? s / n : 0; Y.rows = ['loss']; Y.cols = [''];
  return tp.op(Object.assign({
    type: 'nll', ins: [Pr, Tg], out: Y,
    back(g) {
      const gP = new Float64Array(Pr.v.length);
      for (let q = 0; q < gP.length; q++) if (Tg.v[q]) gP[q] = -g[0] / (n * Math.max(Pr.v[q], 1e-12));
      return [gP, null];
    },
    deps() {
      const cells = []; for (let q = 0; q < Tg.v.length; q++) if (Tg.v[q]) cells.push([Math.floor(q / Tg.c), q % Tg.c]);
      return [{ t: 0, cells }, { t: 1, cells }];
    }
  }, meta));
}

/* ---------- data ---------- */
function makeExample(content) {
  const n = content.length, rev = content.slice().reverse();
  const src = Array(LS - n - 1).fill(PAD).concat([BOS], content);
  const decIn = [BOS].concat(rev); while (decIn.length < LT) decIn.push(PAD);
  const tgt = rev.concat([EOS]); while (tgt.length < LT) tgt.push(PAD);
  const w = tgt.map((_, i) => i <= n ? 1 : 0);
  return { content, src, decIn, tgt, w };
}
function randomExample(rng) {
  const n = 1 + Math.floor(rng() * MAXN);
  return makeExample(Array.from({ length: n }, () => 3 + Math.floor(rng() * LETTERS.length)));
}
function parseContent(text) {
  const toks = text.toLowerCase().replace(/[^a-e]/g, '').split('');
  if (toks.length < 1 || toks.length > MAXN) return null;
  return toks.map(t => VOCAB.indexOf(t));
}

/* ---------- constant matrices ---------- */
function onehot(ids, w) {
  const v = new Float64Array(ids.length * V);
  ids.forEach((t, i) => { if (!w || w[i]) v[i * V + t] = 1; });
  return v;
}
function posEnc(L) {
  const v = new Float64Array(L * D);
  for (let p = 0; p < L; p++) for (let i = 0; i < D / 2; i++) {
    const a = p / Math.pow(10000, 2 * i / D);
    v[p * D + 2 * i] = Math.sin(a); v[p * D + 2 * i + 1] = Math.cos(a);
  }
  return v;
}
const lab = ids => ids.map(t => VOCAB[t]);
const posLab = L => Array.from({ length: L }, (_, i) => 'pos ' + i);

/* ---------- the model ---------- */
function attention(tp, P, pre, Xq, Xkv, M, grp, kind) {
  const prm = (n, name) => tp.param(P, pre + '.' + n, { name, rows: null });
  const q = linear(tp, Xq, prm('Wq', 'W_Q'), null, { key: pre + '.q', group: grp, title: 'Queries', name: 'Q',
    tex: 'Q = X' + (kind === 'cross' ? '_{\\text{dec}}' : '') + 'W_Q' });
  const k = linear(tp, Xkv, prm('Wk', 'W_K'), null, { key: pre + '.k', group: grp, title: 'Keys', name: 'K',
    tex: 'K = ' + (kind === 'cross' ? 'M' : 'X') + 'W_K' });
  const v = linear(tp, Xkv, prm('Wv', 'W_V'), null, { key: pre + '.v', group: grp, title: 'Values', name: 'V',
    tex: 'V = ' + (kind === 'cross' ? 'M' : 'X') + 'W_V' });
  const s = scores(tp, q, k, { key: pre + '.s', group: grp, title: 'Scores', name: 'S', tex: 'S = QK^\\top/\\sqrt{d}' });
  const sm = mask(tp, s, M, { key: pre + '.mask', group: grp, title: kind === 'causal' ? 'Causal mask' : 'Padding mask',
    name: 'S′', tex: "S'_{ij} = \\begin{cases} S_{ij} & \\text{allowed} \\\\ -\\infty & \\text{blocked} \\end{cases}" });
  const a = softmax(tp, sm, { key: pre + '.attn', group: grp, title: 'Attention weights', name: 'A', tex: "A = \\operatorname{softmax}(S')", isAttn: true });
  const h = linear(tp, a, v, null, { key: pre + '.av', group: grp, title: 'Weighted sum of values', name: 'H', tex: 'H = AV' });
  const o = linear(tp, h, prm('Wo', 'W_O'), null, { key: pre + '.o', group: grp, title: 'Output projection', name: 'O', tex: 'O = HW_O' });
  return { q, k, v, s, sm, a, h, o };
}
function addNorm(tp, P, X, Y, lnName, grp, outName) {
  const r = add(tp, X, Y, { key: lnName + '.res', group: grp, title: 'Residual add', name: 'R', tex: 'R = X + \\text{sublayer}(X)' });
  return layernorm(tp, r, tp.param(P, lnName + '.g', { name: 'γ' }), tp.param(P, lnName + '.b', { name: 'β' }),
    { key: lnName + '.ln', group: grp, title: 'Layer norm', name: outName, tex: '\\text{LN}(R)_{ij} = \\gamma_j\\,\\frac{R_{ij}-\\mu_i}{\\sigma_i} + \\beta_j' });
}
function ffn(tp, P, pre, X, grp) {
  const prm = (n, name) => tp.param(P, pre + '.' + n, { name });
  const h1 = linear(tp, X, prm('W1', 'W_1'), prm('b1', 'b_1'), { key: pre + '.l1', group: grp, title: 'Expand to d_ff = 16', name: 'U', tex: 'U = XW_1 + b_1' });
  const h2 = relu(tp, h1, { key: pre + '.relu', group: grp, title: 'ReLU', name: 'U⁺', tex: 'U^+ = \\max(0, U)' });
  return linear(tp, h2, prm('W2', 'W_2'), prm('b2', 'b_2'), { key: pre + '.l2', group: grp, title: 'Project back to d = 8', name: 'F', tex: 'F = U^+W_2 + b_2' });
}

function forward(P, ex) {
  const tp = new Tape();
  const srcL = lab(ex.src), decL = lab(ex.decIn);
  // ----- encoder -----
  const ohS = tp.konst(LS, V, onehot(ex.src), { name: 'source one-hot', rows: srcL, cols: VOCAB, look: 'onehot' });
  const E1 = tp.param(P, 'E', { name: 'E', rows: VOCAB });
  const xe = embed(tp, ohS, E1, ex.src, { key: 'enc.emb', group: 'enc.in', title: 'Embedding lookup', name: 'X_emb', tex: 'X_{\\text{emb}} = \\text{onehot}(\\text{src})\\,E' });
  const pe1 = tp.konst(LS, D, posEnc(LS), { name: 'PE', rows: posLab(LS) });
  const x0 = add(tp, xe, pe1, { key: 'enc.pe', group: 'enc.in', title: 'Add positional encoding', name: 'X', tex: 'X = X_{\\text{emb}} + PE' });
  const padM = new Float64Array(LS * LS);
  for (let i = 0; i < LS; i++) for (let j = 0; j < LS; j++) padM[i * LS + j] = ex.src[j] !== PAD ? 1 : 0;
  const mEnc = tp.konst(LS, LS, padM, { name: 'padding mask', rows: srcL, cols: srcL, look: 'mask' });
  const ea = attention(tp, P, 'enc.self', x0, x0, mEnc, 'enc.attn', 'pad');
  const n1 = addNorm(tp, P, x0, ea.o, 'enc.ln1', 'enc.an1', 'N₁');
  const ef = ffn(tp, P, 'enc.ffn', n1, 'enc.ffn');
  const M = addNorm(tp, P, n1, ef, 'enc.ln2', 'enc.an2', 'M');
  // ----- decoder -----
  const ohT = tp.konst(LT, V, onehot(ex.decIn), { name: 'decoder-input one-hot', rows: decL, cols: VOCAB, look: 'onehot' });
  const E2 = tp.param(P, 'E', { name: 'E', rows: VOCAB });
  const ye = embed(tp, ohT, E2, ex.decIn, { key: 'dec.emb', group: 'dec.in', title: 'Embedding lookup (same E)', name: 'Y_emb', tex: 'Y_{\\text{emb}} = \\text{onehot}(\\text{dec in})\\,E' });
  const pe2 = tp.konst(LT, D, posEnc(LT), { name: 'PE', rows: posLab(LT) });
  const y0 = add(tp, ye, pe2, { key: 'dec.pe', group: 'dec.in', title: 'Add positional encoding', name: 'Y', tex: 'Y = Y_{\\text{emb}} + PE' });
  const cau = new Float64Array(LT * LT);
  for (let i = 0; i < LT; i++) for (let j = 0; j < LT; j++) cau[i * LT + j] = j <= i ? 1 : 0;
  const mCau = tp.konst(LT, LT, cau, { name: 'causal mask', rows: decL, cols: decL, look: 'mask' });
  const ds = attention(tp, P, 'dec.self', y0, y0, mCau, 'dec.self', 'causal');
  const d1 = addNorm(tp, P, y0, ds.o, 'dec.ln1', 'dec.an1', 'D₁');
  const crossM = new Float64Array(LT * LS);
  for (let i = 0; i < LT; i++) for (let j = 0; j < LS; j++) crossM[i * LS + j] = ex.src[j] !== PAD ? 1 : 0;
  const mCross = tp.konst(LT, LS, crossM, { name: 'padding mask', rows: decL, cols: srcL, look: 'mask' });
  const dc = attention(tp, P, 'dec.cross', d1, M, mCross, 'dec.cross', 'cross');
  const d2 = addNorm(tp, P, d1, dc.o, 'dec.ln2', 'dec.an2', 'D₂');
  const df = ffn(tp, P, 'dec.ffn', d2, 'dec.ffn');
  const d3 = addNorm(tp, P, d2, df, 'dec.ln3', 'dec.an3', 'D₃');
  // ----- output -----
  const logits = linear(tp, d3, tp.param(P, 'out.W', { name: 'W_out', cols: VOCAB }), tp.param(P, 'out.b', { name: 'b_out', cols: VOCAB }),
    { key: 'out.logits', group: 'dec.out', title: 'Logits over the vocabulary', name: 'Z', cols: VOCAB, tex: 'Z = D_3W_{\\text{out}} + b_{\\text{out}}' });
  const probs = softmax(tp, logits, { key: 'out.probs', group: 'dec.out', title: 'Next-token probabilities', name: 'P', tex: 'P = \\operatorname{softmax}(Z)' });
  const tg = tp.konst(LT, V, onehot(ex.tgt, ex.w), { name: 'targets', rows: lab(ex.tgt).map((t, i) => ex.w[i] ? t : '—'), cols: VOCAB, look: 'onehot' });
  const loss = nll(tp, probs, tg, { key: 'out.loss', group: 'dec.out', title: 'Cross-entropy loss', name: 'L',
    tex: 'L = -\\tfrac{1}{n}\\sum_t \\log P_{t,\\,y_t}' });
  return { tp, loss, probs, attn: { enc: ea.a, self: ds.a, cross: dc.a } };
}

function argmaxRow(t, i) { let b = 0; for (let j = 1; j < t.c; j++) if (t.v[i * t.c + j] > t.v[i * t.c + b]) b = j; return b; }

/* Greedy generation: feed the model its own previous outputs. */
function generate(P, content) {
  const out = [];
  for (let t = 0; t < LT; t++) {
    const ex = makeExample(content);
    ex.decIn = [BOS].concat(out); while (ex.decIn.length < LT) ex.decIn.push(PAD);
    ex.w = ex.w.map(() => 0);
    const tok = argmaxRow(forward(P, ex).probs, t);
    out.push(tok);
    if (tok === EOS) break;
  }
  return out;
}

/* ---------- training ---------- */
function trainStep(P, st, rng, batch, lr) {
  for (const k in P) P[k].g.fill(0);
  let L = 0, ok = 0, n = 0;
  for (let b = 0; b < batch; b++) {
    const ex = randomExample(rng), f = forward(P, ex);
    f.tp.backward(f.loss, P);
    L += f.loss.v[0];
    for (let i = 0; i < LT; i++) if (ex.w[i]) { n++; if (argmaxRow(f.probs, i) === ex.tgt[i]) ok++; }
  }
  let norm = 0;
  for (const k in P) for (let q = 0; q < P[k].g.length; q++) { P[k].g[q] /= batch; norm += P[k].g[q] ** 2; }
  norm = Math.sqrt(norm);
  const clip = norm > 1 ? 1 / norm : 1;
  st.t = (st.t || 0) + 1;
  const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** st.t, c2 = 1 - b2 ** st.t;
  for (const k in P) {
    const p = P[k];
    for (let q = 0; q < p.v.length; q++) {
      const g = p.g[q] * clip;
      p.m[q] = b1 * p.m[q] + (1 - b1) * g; p.s[q] = b2 * p.s[q] + (1 - b2) * g * g;
      p.v[q] -= lr * (p.m[q] / c1) / (Math.sqrt(p.s[q] / c2) + 1e-8);
    }
  }
  return { loss: L / batch, acc: ok / n };
}

root.TF = { VOCAB, PAD, BOS, EOS, LETTERS, MAXN, LS, LT, D, F, V,
  mulberry32, makeParams, snapshot, fromSnapshot, forward, generate, trainStep,
  makeExample, randomExample, parseContent, argmaxRow };
})(typeof window !== 'undefined' ? window : globalThis);
