// Token Radar v2 — dashboard logic. No network, no libraries: the page
// reads window.RADAR (data.js, written by main.ts) and renders from it.
const D = window.RADAR;
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt$ = (v) => '$' + Math.round(v).toLocaleString('en-US');
const fmtTvl = (v) => v >= 1e9 ? '$' + (v / 1e9).toFixed(2) + 'B' : v >= 1e6 ? '$' + (v / 1e6).toFixed(1) + 'M' : fmt$(v);
const fmtAge = (d) => d === null || d === undefined ? '—' : d >= 365 ? (d / 365.25).toFixed(1) + 'y' : d + 'd';
const fmtChg = (v) => {
  if (v === null || v === undefined) return '—';
  const c = v >= 0 ? 'up' : 'down';
  return `<span class="${c}">${v >= 0 ? '+' : ''}${v.toFixed(1)}%</span>`;
};

const state = { tab: 'board', sortK: 'rank', sortD: 1, q: '', cat: '', chain: '', sb: { g: 1, a: 1, m: 1 }, sel: null };
const COEF = D.model.coefficients;

// The fitted value of one row, with the sandbox weights applied.
// contrib holds each term's log-points at the fitted weights, so the
// sandbox re-weights the age term and scales the total. Growth was
// tested and dropped from the fit (coefficient 0), so the growth
// slider has no effect.
function impliedOf(r) {
  const c = r.contrib;
  const logV = c.base + c.tvl + c.growth * state.sb.g + c.age * state.sb.a + c.chains + c.category;
  return Math.max(1, Math.round(Math.exp(logV) * state.sb.m));
}
const sandboxActive = () => state.sb.g !== 1 || state.sb.a !== 1 || state.sb.m !== 1;

// ---- model panel: the fit, in the open, warts included ----
function renderModel() {
  const m = D.model;
  const coefRows = Object.entries(COEF)
    .map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v >= 0 ? '+' : ''}${v.toFixed(4)}</td></tr>`).join('');
  const wf = m.walk_forward.map((w) =>
    `<tr><td>window from ${esc(w.test_from)} (train ${w.train}, test ${w.test})</td><td>${w.oos_r2.toFixed(3)}</td></tr>`).join('');
  $('#model').innerHTML = `
    <h2>Value model — fitted, not hand-weighted</h2>
    <p class="hint">${esc(m.spec)} on ${m.n_comps} live protocols (a token that already trades, market cap &gt; 0). ${esc(m.winsor)}.</p>
    <div style="display:flex;flex-wrap:wrap;gap:8px 40px;align-items:flex-start">
    <table class="coef"><tr><th>Coefficient</th><th>log points</th></tr>${coefRows}</table>
    <table class="coef"><tr><th>Check</th><th>Value</th></tr>
      <tr><td>R² (in-sample)</td><td>${m.r2.toFixed(3)}</td></tr>
      <tr><td>Adjusted R²</td><td>${m.adj_r2.toFixed(3)}</td></tr>
      <tr><td>Residual SD (log)</td><td>${m.resid_sd_log.toFixed(3)}</td></tr>
      <tr><td>Residual 5–95% (log)</td><td>${m.resid_pct_log[0].toFixed(2)} to ${m.resid_pct_log[4].toFixed(2)}</td></tr>
      <tr><td>5-fold CV, out-of-sample R²</td><td>${m.cv_5fold.oos_r2.toFixed(3)}</td></tr>
      <tr><td>5-fold CV, RMSE (log)</td><td>${m.cv_5fold.rmse_log.toFixed(3)}</td></tr>
      <tr><td>Placebo refit R² (shuffled target)</td><td>${m.placebo.r2.toFixed(3)}</td></tr>
    </table>
    <table class="coef"><tr><th>Walk-forward, out-of-sample R²</th><th></th></tr>${wf}</table>
    <table class="coef"><tr><th>Unconstrained fit — raw growth, shown for comparison</th><th>Value</th></tr>
      <tr><td>Growth coefficient (log points)</td><td>${m.unconstrained.growth_coef.toFixed(4)}</td></tr>
      <tr><td>R²</td><td>${m.unconstrained.r2.toFixed(3)}</td></tr>
    </table>
    </div>
    ${m.growth_dropped ? '<p class="note"><b>Growth dropped:</b> the growth_pos refit still gave a non-positive growth coefficient, so growth was tested and dropped from this fit (coefficient 0) — see the honest-read note below.</p>' : ''}
    <p class="note"><b>Read honestly:</b> ${esc(m.note)}</p>`;
}

// ---- sandbox ----
function bindSandbox() {
  const map = [['sbGrowth', 'g', 'sbGrowthV'], ['sbAge', 'a', 'sbAgeV'], ['sbMult', 'm', 'sbMultV']];
  for (const [id, key, vid] of map) {
    $('#' + id).addEventListener('input', (e) => {
      state.sb[key] = parseFloat(e.target.value);
      $('#' + vid).textContent = state.sb[key].toFixed(2) + '×';
      renderTable();
    });
  }
  $('#sbReset').addEventListener('click', () => {
    state.sb = { g: 1, a: 1, m: 1 };
    for (const [id, , vid] of map) { $('#' + id).value = id === 'sbMult' ? '1' : '1'; $('#' + vid).textContent = '1.00×'; }
    renderTable();
  });
}

// ---- table ----
function viewRows() {
  let rows;
  if (state.tab === 'live') {
    rows = D.live.map((p) => ({ ...p, implied: p.mcap, rank: 0 }));
    rows.sort((a, b) => b.mcap - a.mcap || (a.slug < b.slug ? -1 : 1));
    rows.forEach((r, i) => { r.rank = i + 1; });
  } else {
    rows = D.rows.filter((r) => state.tab === 'board' || r.watchlist);
    rows = rows.map((r) => ({ ...r, implied: impliedOf(r) }));
    rows.sort((a, b) => b.implied - a.implied || (a.slug < b.slug ? -1 : 1));
    rows.forEach((r, i) => { r.drank = i + 1; });
  }
  const q = state.q.trim().toLowerCase();
  if (q) rows = rows.filter((r) => (r.name + ' ' + r.slug).toLowerCase().includes(q));
  if (state.cat) rows = rows.filter((r) => r.category === state.cat);
  if (state.chain) rows = rows.filter((r) => r.chains.includes(state.chain));
  const k = state.sortK;
  const dir = state.sortD;
  const val = (r) => (k === 'rank' ? (r.drank ?? r.rank) : k === 'implied' ? r.implied : k === 'chains' ? r.chains[0] : r[k]);
  rows = [...rows].sort((a, b) => {
    const va = val(a), vb = val(b);
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    return (va < vb ? -1 : va > vb ? 1 : (a.slug < b.slug ? -1 : 1)) * dir;
  });
  return rows;
}

function renderTable() {
  const rows = viewRows();
  const live = state.tab === 'live';
  const lastTh = document.querySelector('#board thead th:last-child');
  lastTh.textContent = live ? 'Market cap' : 'Implied value';
  lastTh.dataset.k = 'implied';
  $('#count').textContent =
    `${rows.length} row${rows.length === 1 ? '' : 's'}` +
    (state.tab === 'board' ? ` · ${D.rows.filter((r) => r.watchlist).length} watchlist · median ${fmt$(D.meta.implied_median)}` : '') +
    (sandboxActive() && !live ? ' · sandbox re-rank active' : '');
  $('#board tbody').innerHTML = rows.map((r) => `<tr data-slug="${esc(r.slug)}">
    <td class="rank">${live ? r.rank : r.drank ?? r.rank}</td>
    <td>${esc(r.name)}${r.watchlist ? '<span class="badge">WATCH</span>' : ''}</td>
    <td>${esc(r.category ?? '—')}</td>
    <td>${esc(r.chains.slice(0, 2).join(', '))}${r.chains.length > 2 ? ' +' + (r.chains.length - 2) : ''}</td>
    <td class="num">${fmtTvl(r.tvl)}</td>
    <td class="num">${fmtChg(r.change_7d)}</td>
    <td class="num">${fmtAge(r.age_days)}</td>
    <td class="num">${fmt$(r.implied)}</td></tr>`).join('');
  document.querySelectorAll('#board tbody tr').forEach((tr) =>
    tr.addEventListener('click', () => openDetail(tr.dataset.slug)));
}

// ---- detail drawer ----
function openDetail(slug) {
  const r = D.rows.find((x) => x.slug === slug) || D.live.find((x) => x.slug === slug);
  if (!r) return;
  state.sel = slug;
  const live = r.mcap !== undefined;
  const c = r.contrib;
  const terms = [
    ['Intercept', c ? c.base : 0], ['log TVL', c ? c.tvl : 0], ['Growth 7d', c ? c.growth : 0],
    ['Age', c ? c.age : 0], ['Chains', c ? c.chains : 0], ['Category', c ? c.category : 0],
  ];
  const maxAbs = Math.max(0.01, ...terms.map(([, v]) => Math.abs(v)));
  const bars = terms.map(([k, v]) => {
    const w = Math.abs(v) / maxAbs * 100;
    return `<div class="row"><span>${k}</span><div class="bar"><i style="width:${w.toFixed(0)}%"></i></div><span>${v >= 0 ? '+' : ''}${v.toFixed(2)}</span></div>`;
  }).join('');
  const farm = r.farm_steps
    ? `<h3>Farm plan — 7 steps</h3><ol class="farm">${r.farm_steps.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>`
    : `<h3>Live token</h3><p class="note">This protocol's token already trades, so it calibrates the model instead of being ranked. Market cap ${fmt$(r.mcap)} on TVL ${fmtTvl(r.tvl)} — a multiple of ${(r.mcap / r.tvl).toFixed(3)}×.</p>`;
  $('#drawer').innerHTML = `
    <button type="button" class="close" id="dClose">Close ✕</button>
    <h2>${esc(r.name)}</h2>
    <p class="meta">${esc(r.category ?? '—')} · ${esc(r.chains.join(', '))} · ${r.watchlist ? 'watchlist' : live ? 'live token' : 'early'}</p>
    <div class="kv">
      <div>TVL<b>${fmtTvl(r.tvl)}</b></div>
      <div>7-day change<b>${r.change_7d === null ? '—' : r.change_7d.toFixed(1) + '%'}</b></div>
      <div>Age<b>${fmtAge(r.age_days)}</b></div>
      <div>${live ? 'Market cap' : 'Implied value'}<b>${fmt$(live ? r.mcap : impliedOf(r))}</b></div>
      ${live ? '' : `<div>Rank<b>#${r.rank} of ${D.rows.length}</b></div><div>Model log value<b>${(c.base + c.tvl + c.growth + c.age + c.chains + c.category).toFixed(2)}</b></div>`}
    </div>
    ${live ? '' : `<h3>Model breakdown (log points)</h3><div class="break">${bars}</div>
    <p class="note">log implied = ${COEF.const.toFixed(2)} + ${COEF.log_tvl.toFixed(2)}·log TVL ${COEF.growth_pos < 0 ? '−' : '+'} ${Math.abs(COEF.growth_pos).toFixed(2)}·growth + ${COEF.age_years.toFixed(2)}·age + ${COEF.chains.toFixed(2)}·chains + category. The sandbox sliders re-weight the age term and scale the total. Growth was tested and dropped from the fit (coefficient 0), so the growth slider has no effect.</p>`}
    ${farm}`;
  $('#drawer').hidden = false;
  $('#dClose').addEventListener('click', closeDetail);
  history.replaceState(null, '', '#p=' + slug);
}
function closeDetail() {
  $('#drawer').hidden = true;
  state.sel = null;
  history.replaceState(null, '', location.pathname + location.search);
}

// ---- init ----
function init() {
  const cats = [...new Set(D.rows.map((r) => r.category))].filter(Boolean).sort();
  const chains = [...new Set(D.rows.flatMap((r) => r.chains))].sort();
  $('#fCat').innerHTML += cats.map((c) => `<option>${esc(c)}</option>`).join('');
  $('#fChain').innerHTML += chains.map((c) => `<option>${esc(c)}</option>`).join('');
  $('#stats').innerHTML =
    `<div>Board <b>${D.rows.length} rows</b></div><div>Watchlist <b>${D.rows.filter((r) => r.watchlist).length}/${D.meta.watchlist_total}</b></div>` +
    `<div>Live comps <b>${D.live.length}</b></div><div>Median implied <b>${fmt$(D.meta.implied_median)}</b></div>` +
    `<div>Total implied <b>${fmt$(D.meta.implied_total)}</b></div><div>Model R² <b>${D.model.r2.toFixed(3)}</b></div>` +
    `<div>Snapshot <b>${esc(D.meta.as_of)}</b></div>`;
  renderModel();
  bindSandbox();
  $('#q').addEventListener('input', (e) => { state.q = e.target.value; renderTable(); });
  $('#fCat').addEventListener('change', (e) => { state.cat = e.target.value; renderTable(); });
  $('#fChain').addEventListener('change', (e) => { state.chain = e.target.value; renderTable(); });
  document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((x) => x.classList.remove('on'));
    b.classList.add('on');
    state.tab = b.dataset.tab;
    renderTable();
  }));
  document.querySelectorAll('#board thead th').forEach((th) => th.addEventListener('click', () => {
    const k = th.dataset.k;
    if (state.sortK === k) state.sortD = -state.sortD;
    else { state.sortK = k; state.sortD = k === 'name' || k === 'category' || k === 'chains' ? 1 : -1; }
    renderTable();
  }));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDetail(); });
  renderTable();
  const m = location.hash.match(/#p=([\w-]+)/);
  if (m) openDetail(m[1]);
}
init();
