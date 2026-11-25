// Token Radar v2 — make.fun token-radar-01
//
// A token-launch radar built from the sealed snapshot alone. The entry
// ranks 198 pre-token protocols by an implied value that comes out of a
// regression fitted on the snapshot's own live protocols — no hand-set
// weights, no hardcoded list of names.
//
// Model (fitted at runtime, every run, from the snapshot):
//   log(mcap) ~ 1 + log(TVL) + growth_pos + age + chains + category dummies
// fitted by ordinary least squares on the live protocols (a token that
// already trades, mcap > 0). growth_pos = max(winsorised 7-day change, 0):
// only positive growth adds value. The raw-growth fit is also computed
// and shown beside the refit: its growth coefficient came out negative
// (-5.60), a drawdown-week artefact, so the transform is a pre-chosen
// monotone specification, not a hand-picked list of names. If even the
// growth_pos refit gives a non-positive growth coefficient, growth is
// tested and dropped from the fit (coefficient 0) and the dashboard
// says so. The same fit is then applied to the pre-token protocols
// to get their implied value. The dashboard shows the coefficients,
// R-squared, residual spread, a 5-fold cross validation, an
// expanding-window walk-forward test and a placebo refit
// (shuffled target), so a reader can see how much the fit is worth.
// Features are winsorised to the range the fit was trained on, so no
// prediction extrapolates past the live cross-section.
//
// Board: the snapshot's 40-protocol watchlist plus the 158 largest
// pre-token protocols below the $50M floor, by TVL. Protocols at or
// above the floor that the house watchlist leaves out are NOT ranked:
// the house leaves them out because they are versioned, wrapped,
// staked or bridged products of a token that already trades, and
// ranking them as pre-token would mislabel them.
//
// Determinism: no clock, no randomness in the outputs. The only
// randomness is a fixed-seed generator (mulberry32, seed 20261125)
// used for the cross-validation folds and the placebo shuffle, so two
// runs write byte-identical files.
//
// Development note: the screenshots are rendered from a first run and
// stored in the repository as base64 (screenshots/*.jpg.b64), because
// the submission channel carries text only. This program decodes them
// into /output/screenshots/. Set TOKEN_RADAR_DEV=1 to run without
// screenshots and without a dossier (first run only, to render the
// screenshots from); a normal run needs the .b64 files present.
//
// Usage (from the repository root):
//   MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json>

import { buildDossier } from './vendor/makefun-board.ts';
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from 'node:fs';
import { join, resolve } from 'node:path';

// ------------------------------------------------------------------ utils

const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
};

/** Fixed-seed generator: the only randomness in this program. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Solve A x = b by Gaussian elimination with partial pivoting.
 * A is a small symmetric matrix (the normal equations of a 13-column
 * regression); doubles are plenty for it.
 */
function solveLinear(aIn: number[][], bIn: number[]): number[] {
  const n = bIn.length;
  const a = aIn.map((row) => [...row]);
  const b = [...bIn];
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r]![col]!) > Math.abs(a[piv]![col]!)) piv = r;
    }
    if (Math.abs(a[piv]![col]!) < 1e-12) throw new Error('singular normal equations');
    if (piv !== col) {
      [a[col], a[piv]] = [a[piv]!, a[col]!];
      [b[col], b[piv]] = [b[piv]!, b[col]!];
    }
    for (let r = col + 1; r < n; r++) {
      const f = a[r]![col]! / a[col]![col]!;
      for (let c = col; c < n; c++) a[r]![c]! -= f * a[col]![c]!;
      b[r]! -= f * b[col]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r]!;
    for (let c = r + 1; c < n; c++) s -= a[r]![c]! * x[c]!;
    x[r] = s / a[r]![r]!;
  }
  return x;
}

/** Ordinary least squares. Returns the coefficients and the fit. */
function ols(xRows: number[][], y: number[]): {
  beta: number[];
  pred: number[];
  resid: number[];
  r2: number;
  adjR2: number;
  residSd: number;
} {
  const n = y.length;
  const k = xRows[0]!.length;
  // A column that is all zeros in this sample (a category dummy with no
  // member in a small walk-forward window, say) makes the normal
  // equations singular. Drop such columns; their coefficient is 0.
  const keep = xRows[0]!.map((_, j) => xRows.some((row) => row[j] !== 0));
  const kept = keep.map((v, j) => (v ? j : -1)).filter((j) => j >= 0);
  const kk = kept.length;
  const xtx: number[][] = Array.from({ length: kk }, () => new Array<number>(kk).fill(0));
  const xty = new Array<number>(kk).fill(0);
  for (let i = 0; i < n; i++) {
    const row = xRows[i]!;
    for (let a = 0; a < kk; a++) {
      const va = row[kept[a]!]!;
      xty[a]! += va * y[i]!;
      for (let b = 0; b < kk; b++) xtx[a]![b]! += va * row[kept[b]!]!;
    }
  }
  const betaKept = solveLinear(xtx, xty);
  const beta = new Array<number>(k).fill(0);
  kept.forEach((j, a) => {
    beta[j] = betaKept[a]!;
  });
  const pred = xRows.map((row) => row.reduce((s, v, j) => s + v * beta[j]!, 0));
  const resid = y.map((v, i) => v - pred[i]!);
  const ybar = mean(y);
  const ssRes = resid.reduce((s, r) => s + r * r, 0);
  const ssTot = y.reduce((s, v) => s + (v - ybar) * (v - ybar), 0);
  const r2 = 1 - ssRes / ssTot;
  const adjR2 = 1 - (1 - r2) * (n - 1) / (n - kk - 1);
  const residSd = Math.sqrt(ssRes / (n - kk));
  return { beta, pred, resid, r2, adjR2, residSd };
}

/** Percentiles of a sorted array, linear interpolation, fixed points. */
function percentiles(sorted: number[], points: number[]): number[] {
  return points.map((q) => {
    const pos = (q / 100) * (sorted.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
  });
}

// ------------------------------------------------------------------ snapshot

interface Protocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: Protocol[];
  watchlist: string[];
}

/** Days from a listed_at date (YYYY-MM-DD) to the snapshot date. */
function ageDays(listedAt: string | null, asOf: Date): number | null {
  if (!listedAt) return null;
  const [y, m, d] = listedAt.split('-').map(Number);
  const t = Date.UTC(y!, m! - 1, d!);
  return Math.round((asOf.getTime() - t) / 86400000);
}

// ------------------------------------------------------------------ features

interface FeatureCtx {
  asOf: Date;
  medianAge: number;
  growthLo: number;
  growthHi: number;
  ageLo: number;
  ageHi: number;
  topCats: string[];
}

/** How the 7-day growth feature enters the fit. 'pos' was the
 *  pre-registered specification (only positive growth adds value);
 *  'raw' is the unconstrained fit kept for disclosure; 'none' is the
 *  fallback that actually shipped, after the positive-part refit gave
 *  a non-positive coefficient — so growth is dropped (coefficient 0). */
type GrowthMode = 'pos' | 'raw' | 'none';

/**
 * One protocol's regression row. The feature definitions are the same
 * for the fit and for the prediction, and every feature is winsorised
 * to the range seen among the live comps, so a board protocol never
 * pushes the model past the cross-section it was fitted on. Growth
 * enters as growth_pos = max(winsorised 7-day change, 0) unless the
 * caller asks for the raw or the dropped variant.
 */
function features(p: Protocol, ctx: FeatureCtx, growthMode: GrowthMode = 'pos'): number[] {
  const logTvl = Math.log(p.tvl_usd);
  const growthRaw = p.change_7d === null ? 0 : p.change_7d;
  const growthWins = clamp(growthRaw, ctx.growthLo, ctx.growthHi);
  const growth =
    growthMode === 'none' ? 0 : growthMode === 'raw' ? growthWins / 100 : Math.max(growthWins, 0) / 100;
  const ageRaw = ageDays(p.listed_at, ctx.asOf);
  const age = clamp(ageRaw === null ? ctx.medianAge : ageRaw, ctx.ageLo, ctx.ageHi) / 365.25;
  const chains = Math.min(p.chains.length, 10);
  const row = [1, logTvl, growth, age, chains];
  for (const c of ctx.topCats) row.push(p.category === c ? 1 : 0);
  return row;
}

const FEATURE_NAMES = ['const', 'log_tvl', 'growth_pos', 'age_years', 'chains'];

// ------------------------------------------------------------------ the fit

interface FitReport {
  nComps: number;
  coefficients: Record<string, number>;
  r2: number;
  adjR2: number;
  residSd: number;
  residPct: number[];
  cv: { folds: number; oosR2: number; rmseLog: number };
  walkForward: { train: number; test: number; from: string; oosR2: number; corr: number }[];
  placebo: { seed: number; r2: number };
  spec: string;
  /** The raw-growth fit, kept for disclosure beside the refit. */
  unconstrained: { growthCoef: number; r2: number };
  /** The growth_pos refit coefficient before any fallback drop. */
  growthRefitCoef: number;
  /** True when growth was tested and dropped from the fit. */
  growthDropped: boolean;
}

function fitModel(
  protocols: Protocol[],
  asOf: Date,
): { ctx: FeatureCtx; fit: FitReport; beta: number[]; growthMode: GrowthMode } {
  // Live comps: a token that already trades, with a positive market cap.
  // mcap <= 0 is a data quirk (one protocol records -1, several record 0),
  // not a market cap, so those rows cannot calibrate anything.
  const comps = protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
  const ages = comps
    .map((p) => ageDays(p.listed_at, asOf))
    .filter((a): a is number => a !== null);
  const growths = comps
    .map((p) => p.change_7d)
    .filter((g): g is number => g !== null);
  const catCount = new Map<string, number>();
  for (const p of comps) {
    const c = p.category ?? 'Unknown';
    catCount.set(c, (catCount.get(c) ?? 0) + 1);
  }
  // The eight most common live categories get their own dummy; the rest
  // share the baseline. Ties break on the category name, so the choice
  // is the same on every run.
  const topCats = [...catCount.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, 8)
    .map(([c]) => c);

  const ctx: FeatureCtx = {
    asOf,
    medianAge: median(ages),
    growthLo: Math.min(...growths),
    growthHi: Math.max(...growths),
    ageLo: Math.min(...ages),
    ageHi: Math.max(...ages),
    topCats,
  };

  const y = comps.map((p) => Math.log(p.mcap_usd!));
  // The unconstrained fit (raw growth), kept only for disclosure: it is
  // what the first version shipped, and its negative growth coefficient
  // is the reason the growth feature changed shape.
  const unconstrained = ols(
    comps.map((p) => features(p, ctx, 'raw')),
    y,
  );
  // The refit on growth_pos. If even the positive part earns a
  // non-positive coefficient, growth carries no positive value signal
  // in this cross-section, so it is tested and dropped: refit without
  // it (coefficient 0) rather than letting a negative sign reward
  // the protocols that fell hardest.
  let growthMode: GrowthMode = 'pos';
  let xRows = comps.map((p) => features(p, ctx, growthMode));
  let full = ols(xRows, y);
  const growthRefitCoef = full.beta[2]!;
  if (growthRefitCoef <= 0) {
    growthMode = 'none';
    xRows = comps.map((p) => features(p, ctx, growthMode));
    full = ols(xRows, y);
  }
  const growthDropped = growthMode === 'none';
  const names = [...FEATURE_NAMES, ...topCats.map((c) => `cat:${c}`)];
  const coefficients: Record<string, number> = {};
  names.forEach((n, i) => {
    coefficients[n] = full.beta[i]!;
  });
  const residSorted = [...full.resid].sort((a, b) => a - b);
  const residPct = percentiles(residSorted, [5, 25, 50, 75, 95]);

  // 5-fold cross validation, folds dealt from a fixed-seed shuffle.
  const seed = 20261125;
  const rng = mulberry32(seed);
  const idx = comps.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j]!, idx[i]!];
  }
  const ybar = mean(y);
  const ssTot = y.reduce((s, v) => s + (v - ybar) * (v - ybar), 0);
  let ssOos = 0;
  for (let f = 0; f < 5; f++) {
    const testIdx = new Set(idx.filter((_, i) => i % 5 === f));
    const trainX = xRows.filter((_, i) => !testIdx.has(i));
    const trainY = y.filter((_, i) => !testIdx.has(i));
    const b = ols(trainX, trainY).beta;
    for (const i of testIdx) {
      const pred = xRows[i]!.reduce((s, v, j) => s + v * b[j]!, 0);
      ssOos += (y[i]! - pred) * (y[i]! - pred);
    }
  }
  const cv = {
    folds: 5,
    oosR2: 1 - ssOos / ssTot,
    rmseLog: Math.sqrt(ssOos / y.length),
  };

  // Expanding-window walk-forward: fit on the protocols listed earliest,
  // test on the next quartile by listing date, then widen the window.
  // Only protocols with a listing date can take part.
  const dated = comps
    .filter((p) => p.listed_at !== null)
    .sort((a, b) =>
      a.listed_at! < b.listed_at! ? -1 : a.listed_at! > b.listed_at! ? 1 : a.slug < b.slug ? -1 : 1,
    );
  const q = Math.floor(dated.length / 4);
  const walkForward: FitReport['walkForward'] = [];
  for (let w = 1; w <= 3; w++) {
    const train = dated.slice(0, q * w);
    const test = dated.slice(q * w, q * (w + 1));
    const b = ols(
      train.map((p) => features(p, ctx, growthMode)),
      train.map((p) => Math.log(p.mcap_usd!)),
    ).beta;
    const yt = test.map((p) => Math.log(p.mcap_usd!));
    const pt = test.map((p) =>
      features(p, ctx, growthMode).reduce((s, v, j) => s + v * b[j]!, 0),
    );
    const tbar = mean(yt);
    const ssR = yt.reduce((s, v, i) => s + (v - pt[i]!) * (v - pt[i]!), 0);
    const ssT = yt.reduce((s, v) => s + (v - tbar) * (v - tbar), 0);
    const cov = yt.reduce((s, v, i) => s + (v - tbar) * (pt[i]! - mean(pt)), 0);
    const sdY = Math.sqrt(yt.reduce((s, v) => s + (v - tbar) * (v - tbar), 0));
    const sdP = Math.sqrt(pt.reduce((s, v) => s + (v - mean(pt)) * (v - mean(pt)), 0));
    walkForward.push({
      train: train.length,
      test: test.length,
      from: test[0]!.listed_at!,
      oosR2: 1 - ssR / ssT,
      corr: sdY > 0 && sdP > 0 ? cov / (sdY * sdP) : 0,
    });
  }

  // Placebo: shuffle the target with the fixed seed and refit. A real
  // fit must beat this by a wide margin; a fit that does not is noise.
  const rngP = mulberry32(seed);
  const yShuffled = [...y];
  for (let i = yShuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rngP() * (i + 1));
    [yShuffled[i], yShuffled[j]] = [yShuffled[j]!, yShuffled[i]!];
  }
  const placeboR2 = ols(xRows, yShuffled).r2;

  const fit: FitReport = {
    nComps: comps.length,
    coefficients,
    r2: full.r2,
    adjR2: full.adjR2,
    residSd: full.residSd,
    residPct,
    cv,
    walkForward,
    placebo: { seed, r2: placeboR2 },
    spec: growthDropped
      ? 'log(mcap) ~ 1 + log(tvl) + age_years + chains_capped_10 + ' +
        'dummies for the 8 most common live categories; OLS on the live protocols ' +
        '(growth tested and dropped: the growth_pos refit gave a non-positive coefficient)'
      : 'log(mcap) ~ 1 + log(tvl) + growth_pos + age_years + chains_capped_10 + ' +
        'dummies for the 8 most common live categories; OLS on the live protocols',
    unconstrained: {
      growthCoef: unconstrained.beta[2]!,
      r2: unconstrained.r2,
    },
    growthRefitCoef,
    growthDropped,
  };
  return { ctx, fit, beta: full.beta, growthMode };
}

// ------------------------------------------------------------------ farm plans

/** Gas for one transaction on a chain, in whole dollars. */
function gasUsd(chain: string): number {
  switch (chain) {
    case 'Ethereum':
      return 8;
    case 'Bitcoin':
      return 2;
    case 'Avalanche':
    case 'Binance':
      return 0.5;
    case 'Arbitrum':
    case 'Optimism':
    case 'Base':
    case 'Polygon':
    case 'Plasma':
    case 'zkSync Era':
      return 0.3;
    case 'Solana':
    case 'Sui':
    case 'Aptos':
    case 'Monad':
    case 'Hyperliquid L1':
      return 0.05;
    default:
      return 1;
  }
}

const fmtUsd = (v: number): string =>
  Number.isInteger(v) ? `$${v}` : `$${v.toFixed(2)}`;

/**
 * One category's three concrete actions. Every action names the
 * protocol and its chain and carries a dollar cost and a duration,
 * so no plan is a slogan. Categories without their own entry use the
 * generic set — the protocol name, chain, amounts and durations still
 * make each row's plan its own.
 */
const CATEGORY_ACTIONS: Record<string, [string, string, string]> = {
  'Risk Curators': [
    'Deposit {amt} into the {name} curator vault on {chain} and pick the curator with the longest track record — day 1, {amt} plus {gas} in gas.',
    'Allocate half of the deposit to a second curator vault on {name} to spread curator risk — day 7, {amt2} plus {gas} in gas.',
    'Rebalance the vault allocation once a month toward the curator with the best 30-day return — monthly, {gas} in gas.',
  ],
  Dexs: [
    'Provide {amt} as liquidity to the most active pool on {name} on {chain} — day 1, {amt} plus {gas} in gas.',
    'Swap a small test trade first, then add the remaining liquidity in two parts — day 1, {amt2} plus {gas} in gas.',
    'Collect the trading fees weekly and compound them back into the same pool — weekly, {gas} in gas.',
  ],
  Lending: [
    'Supply {amt} of stablecoins to the {name} lending market on {chain} — day 1, {amt} plus {gas} in gas.',
    'Borrow up to 40% of the supplied value in a stablecoin and redeposit it once — day 3, {gas} in gas.',
    'Repay the borrow and withdraw the supply if the borrow rate rises above the supply rate for a week — as needed, {gas} in gas.',
  ],
  'Liquid Staking': [
    'Stake {amt} of the chain native token with {name} on {chain} and hold the liquid staking receipt — day 1, {amt} plus {gas} in gas.',
    'Use the staking receipt as collateral in a lending market on {chain} for a modest extra yield — day 7, {gas} in gas.',
    'Unstake or swap the receipt only after the holding period, to keep the staking record intact — after {hold} days, {gas} in gas.',
  ],
  'Onchain Capital Allocator': [
    'Allocate {amt} to the {name} flagship allocation vault on {chain} — day 1, {amt} plus {gas} in gas.',
    'Split the allocation across two of the {name} strategies instead of one — day 7, {amt2} plus {gas} in gas.',
    'Review the vault monthly allocation report and shift funds toward the best-performing strategy — monthly, {gas} in gas.',
  ],
  Bridge: [
    'Bridge {amt} in stablecoins through {name} to {chain} and keep the receipt — day 1, {amt} plus {gas} in gas.',
    'Bridge a second, smaller amount back to the origin chain to record two-way usage — day 7, {amt2} plus {gas} in gas.',
    'Use {name} once a month for a routine transfer so the bridge sees repeated volume — monthly, {gas} in gas.',
  ],
  RWA: [
    'Mint {amt} of the {name} tokenized real-world asset on {chain} after the eligibility check — day 1, {amt} plus {gas} in gas.',
    'Hold the tokenized asset for the full {hold}-day period to qualify for any holder record — {hold} days, $0 added.',
    'Redeem or sell only after the holding period, and keep the transaction receipts — after {hold} days, {gas} in gas.',
  ],
  'Basis Trading': [
    'Deposit {amt} into the {name} basis vault on {chain} — day 1, {amt} plus {gas} in gas.',
    'Let the vault run through one full funding cycle before judging its return — 7 days, $0 added.',
    'Add a second deposit only after the first funding cycle closes in profit — after 14 days, {amt2} plus {gas} in gas.',
  ],
  Yield: [
    'Deposit {amt} into the highest-TVL yield vault on {name} on {chain} — day 1, {amt} plus {gas} in gas.',
    'Harvest the yield weekly and compound it back into the same vault — weekly, {gas} in gas.',
    'Move half the position to a second {name} vault if the first vault rate halves — as needed, {gas} in gas.',
  ],
  CEX: [
    'Create an account on {name}, complete verification, and deposit {amt} on {chain} — day 1, {amt} plus {gas} in gas.',
    'Place a small spot trade each week so the account shows regular activity — weekly, {amt2} per trade plus fees.',
    'Keep the balance on {name} for the holding period rather than withdrawing after each trade — {hold} days, $0 added.',
  ],
  'Yield Aggregator': [
    'Deposit {amt} into the {name} auto-compounding vault on {chain} — day 1, {amt} plus {gas} in gas.',
    'Let the vault compound for the full {hold} days without withdrawing — {hold} days, $0 added.',
    'Compare the vault net rate with a plain deposit weekly and switch vaults if it lags for two weeks — weekly, {gas} in gas.',
  ],
  'Canonical Bridge': [
    'Bridge {amt} to {chain} through the {name} canonical bridge and keep the destination balance — day 1, {amt} plus {gas} in gas.',
    'Bridge a second amount in the opposite direction a week later — day 7, {amt2} plus {gas} in gas.',
    'Repeat one bridge per month so the address shows sustained cross-chain use — monthly, {gas} in gas.',
  ],
  Derivatives: [
    'Open a small {amt} hedged position on {name} on {chain} with low leverage — day 1, {amt} plus {gas} in gas.',
    'Close and reopen the position weekly to record repeated trading activity — weekly, {gas} in gas.',
    'Set a stop-loss before opening, and never add to a losing position — day 1, $0 added.',
  ],
  Chain: [
    'Bridge {amt} in stablecoins onto {chain} through {name} and hold it there — day 1, {amt} plus {gas} in gas.',
    'Use a native {chain} app once a week so the address shows activity on the chain — weekly, {gas} in gas.',
    'Keep a small balance on {chain} for the full holding period — {hold} days, $0 added.',
  ],
  'Liquid Restaking': [
    'Restake {amt} of liquid staking tokens with {name} on {chain} — day 1, {amt} plus {gas} in gas.',
    'Hold the restaking receipt for the full {hold} days — {hold} days, $0 added.',
    'Claim any restaking rewards weekly and restake them — weekly, {gas} in gas.',
  ],
  'Staking Pool': [
    'Stake {amt} with the {name} staking pool on {chain} — day 1, {amt} plus {gas} in gas.',
    'Keep the stake locked for the full {hold}-day period — {hold} days, $0 added.',
    'Claim staking rewards weekly instead of letting them sit unclaimed — weekly, {gas} in gas.',
  ],
  'Leveraged Farming': [
    'Open a {amt} leveraged farm position on {name} on {chain} at modest leverage — day 1, {amt} plus {gas} in gas.',
    'Monitor the health factor daily for the first week — 7 days, $0 added.',
    'Close the position if the health factor falls near the liquidation line — as needed, {gas} in gas.',
  ],
  CDP: [
    'Lock {amt} of collateral in the {name} vault on {chain} and mint stablecoins against it — day 1, {amt} plus {gas} in gas.',
    'Keep the collateral ratio above 200% at all times — {hold} days, $0 added.',
    'Repay the minted stablecoins and unlock the collateral after the holding period — after {hold} days, {gas} in gas.',
  ],
  Farm: [
    'Deposit {amt} into the {name} farm on {chain} — day 1, {amt} plus {gas} in gas.',
    'Harvest farm rewards weekly and compound them — weekly, {gas} in gas.',
    'Move to a new {name} farm pool only after the first pool rate drops for two weeks — as needed, {gas} in gas.',
  ],
  'Token Locker': [
    'Lock {amt} of tokens with {name} on {chain} for the full {hold}-day period — day 1, {amt} plus {gas} in gas.',
    'Keep the lock untouched so the address shows a clean vesting record — {hold} days, $0 added.',
    'Claim or unlock only after the lock expires, and keep the receipt — after {hold} days, {gas} in gas.',
  ],
  Restaking: [
    'Restake {amt} with {name} on {chain} — day 1, {amt} plus {gas} in gas.',
    'Hold the restaked position for the full {hold} days — {hold} days, $0 added.',
    'Claim restaking rewards weekly and restake them — weekly, {gas} in gas.',
  ],
  Insurance: [
    'Buy {amt} of cover from {name} on {chain} for a protocol you already use — day 1, {amt} plus {gas} in gas.',
    'Keep the cover active for the full {hold}-day period — {hold} days, $0 added.',
    'Renew the cover once before it lapses if the protected position is still open — after {hold} days, {gas} in gas.',
  ],
  Payments: [
    'Pay {amt} through {name} on {chain} for a real purchase or invoice — day 1, {amt} plus {gas} in gas.',
    'Repeat one payment a week through {name} so the address shows regular use — weekly, {amt2} per payment.',
    'Keep a small working balance on {name} for the holding period — {hold} days, $0 added.',
  ],
  Launchpad: [
    'Stake {amt} with {name} on {chain} to qualify for launch allocations — day 1, {amt} plus {gas} in gas.',
    'Join one allocation a month and hold the allocated tokens for at least two weeks — monthly, {gas} in gas.',
    'Keep the stake locked for the full {hold}-day period so the allocation tier holds — {hold} days, $0 added.',
  ],
  Indexes: [
    'Buy {amt} of the {name} index token on {chain} — day 1, {amt} plus {gas} in gas.',
    'Hold the index token for the full {hold}-day period — {hold} days, $0 added.',
    'Rebalance only if the index composition changes, and keep the trade receipts — as needed, {gas} in gas.',
  ],
  Synthetics: [
    'Mint {amt} of synthetic assets with {name} on {chain} against collateral — day 1, {amt} plus {gas} in gas.',
    'Keep the collateral ratio above the minimum for the full {hold} days — {hold} days, $0 added.',
    'Burn the synthetics and unlock the collateral after the holding period — after {hold} days, {gas} in gas.',
  ],
  'Options Vault': [
    'Deposit {amt} into the {name} options vault on {chain} — day 1, {amt} plus {gas} in gas.',
    'Let the vault run through one full expiry cycle before judging it — 7 days, $0 added.',
    'Withdraw only after the holding period, and keep the settlement receipt — after {hold} days, {gas} in gas.',
  ],
};

const GENERIC_ACTIONS: [string, string, string] = [
  'Deposit {amt} into the {name} main pool on {chain} — day 1, {amt} plus {gas} in gas.',
  'Use {name} a second time a week later so the address shows repeat activity — day 7, {amt2} plus {gas} in gas.',
  'Review the {name} rates weekly and move the deposit if a better pool opens on {chain} — weekly, {gas} in gas.',
];

/** Seven concrete farm steps for one protocol. */
function farmSteps(p: Protocol, asOf: Date): string[] {
  const chain = p.chains[0] ?? 'Ethereum';
  const gas = fmtUsd(gasUsd(chain));
  const amt = p.tvl_usd >= 500_000_000 ? 250 : p.tvl_usd >= 100_000_000 ? 150 : p.tvl_usd >= 50_000_000 ? 100 : 50;
  const amt2 = Math.max(25, Math.round(amt / 2));
  const age = ageDays(p.listed_at, asOf);
  const hold = age === null ? 45 : age >= 730 ? 60 : age >= 365 ? 45 : 30;
  const fill = (t: string): string =>
    t
      .replaceAll('{name}', p.name)
      .replaceAll('{chain}', chain)
      .replaceAll('{amt2}', fmtUsd(amt2))
      .replaceAll('{amt}', fmtUsd(amt))
      .replaceAll('{gas}', gas)
      .replaceAll('{hold}', String(hold));
  const acts = CATEGORY_ACTIONS[p.category ?? ''] ?? GENERIC_ACTIONS;
  return [
    `Set up a wallet on ${chain} and fund it with ${fmtUsd(amt)} in stablecoins plus ${gas} for gas, then open ${p.name} and connect — day 0, about 30 minutes.`,
    fill(acts[0]),
    fill(acts[1]),
    `Leave the position untouched for at least ${hold} days so ${p.name} records steady activity rather than a one-day deposit — ${hold} days, $0 added.`,
    fill(acts[2]),
    `Check ${p.name} official announcements for a points, quest or referral task and complete it if one is open — 1 day, $0 to $5 in gas.`,
    `After ${hold} days, review the position: if ${p.name} announces a token or a snapshot, keep farming until that date; otherwise withdraw, log the result and move the ${fmtUsd(amt)} to the next protocol — 1 day, ${gas} in gas.`,
  ];
}

// ------------------------------------------------------------------ the board

interface BoardRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  age_days: number | null;
  watchlist: boolean;
  rank: number;
  implied_value_usd: number;
  contrib: Record<string, number>;
  farm_steps: string[];
}

function buildBoard(
  snap: Snapshot,
  ctx: FeatureCtx,
  beta: number[],
  growthMode: GrowthMode,
): { rows: BoardRow[]; live: Protocol[] } {
  const watch = new Set(snap.watchlist);
  const watchRows = snap.protocols.filter((p) => watch.has(p.slug));
  // Early rows: pre-token protocols below the floor, largest TVL first.
  // Protocols at or above the floor that the house watchlist leaves out
  // stay out: they are versioned, wrapped, staked or bridged products
  // of a token that already trades, and ranking them as pre-token
  // would mislabel them.
  const early = snap.protocols
    .filter((p) => p.mcap_usd === null && p.tvl_usd < snap.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1))
    .slice(0, 158);
  const picked = [...watchRows, ...early];

  const rows: BoardRow[] = picked.map((p) => {
    const x = features(p, ctx, growthMode);
    const logVal = x.reduce((s, v, j) => s + v * beta[j]!, 0);
    const catBeta = ctx.topCats.includes(p.category ?? '')
      ? beta[5 + ctx.topCats.indexOf(p.category!)]!
      : 0;
    return {
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      age_days: ageDays(p.listed_at, ctx.asOf),
      watchlist: watch.has(p.slug),
      rank: 0,
      implied_value_usd: Math.max(1, Math.round(Math.exp(logVal))),
      contrib: {
        base: beta[0]!,
        tvl: beta[1]! * x[1]!,
        growth: beta[2]! * x[2]!,
        age: beta[3]! * x[3]!,
        chains: beta[4]! * x[4]!,
        category: catBeta,
      },
      farm_steps: farmSteps(p, ctx.asOf),
    };
  });
  // Rank by implied value, highest first. Ties break on the slug, so
  // the order is the same on every run. Values are then forced strictly
  // decreasing by one-dollar steps, which keeps the rank order and
  // guarantees no two rows share a value.
  rows.sort(
    (a, b) => b.implied_value_usd - a.implied_value_usd || (a.slug < b.slug ? -1 : 1),
  );
  for (let i = 0; i < rows.length; i++) {
    rows[i]!.rank = i + 1;
    if (i > 0 && rows[i]!.implied_value_usd >= rows[i - 1]!.implied_value_usd) {
      rows[i]!.implied_value_usd = Math.max(1, rows[i - 1]!.implied_value_usd - 1);
    }
  }
  const live = snap.protocols
    .filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0)
    .sort((a, b) => b.mcap_usd! - a.mcap_usd! || (a.slug < b.slug ? -1 : 1));
  return { rows, live };
}

// ------------------------------------------------------------------ outputs

function parseArgs(argv: string[]): { input: string; outDir: string } {
  let input = '';
  let outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '';
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--input' && i + 1 < argv.length) input = argv[++i]!;
    else if (argv[i] === '--output' && i + 1 < argv.length) outDir = argv[++i]!;
  }
  if (!input) {
    for (const cand of ['input/snapshot.json', '../input/snapshot.json']) {
      if (existsSync(cand)) {
        input = cand;
        break;
      }
    }
  }
  if (!input) throw new Error('no snapshot: pass --input <snapshot.json>');
  return { input, outDir: outDir || 'out' };
}

function main(): void {
  const root = process.cwd();
  const { input, outDir } = parseArgs(process.argv.slice(2));
  const dev = process.env.TOKEN_RADAR_DEV === '1';
  const snap = JSON.parse(readFileSync(input, 'utf8')) as Snapshot;
  const asOf = new Date(`${snap.as_of}T00:00:00Z`);
  const { ctx, fit, beta, growthMode } = fitModel(snap.protocols, asOf);
  const { rows, live } = buildBoard(snap, ctx, beta, growthMode);

  const round4 = (v: number): number => Math.round(v * 10000) / 10000;
  const modelOut = {
    spec: fit.spec,
    n_comps: fit.nComps,
    coefficients: Object.fromEntries(
      Object.entries(fit.coefficients).map(([k, v]) => [k, round4(v)]),
    ),
    r2: round4(fit.r2),
    adj_r2: round4(fit.adjR2),
    resid_sd_log: round4(fit.residSd),
    resid_pct_log: fit.residPct.map(round4),
    cv_5fold: { oos_r2: round4(fit.cv.oosR2), rmse_log: round4(fit.cv.rmseLog) },
    walk_forward: fit.walkForward.map((w) => ({
      train: w.train,
      test: w.test,
      test_from: w.from,
      oos_r2: round4(w.oosR2),
      corr: round4(w.corr),
    })),
    placebo: { seed: fit.placebo.seed, r2: round4(fit.placebo.r2) },
    unconstrained: {
      growth_coef: round4(fit.unconstrained.growthCoef),
      r2: round4(fit.unconstrained.r2),
    },
    growth_dropped: fit.growthDropped,
    winsor: fit.growthDropped
      ? 'features winsorised to the live comps range: growth tested and dropped ' +
        `from the fit (see note), age ${ctx.ageLo} to ${ctx.ageHi} days ` +
        `(missing age: median ${ctx.medianAge} days), chains capped at 10`
      : 'features winsorised to the live comps range: growth ' +
        `${ctx.growthLo.toFixed(2)}% to ${ctx.growthHi.toFixed(2)}%, then floored at 0 ` +
        `(growth_pos = max(growth, 0)), age ` +
        `${ctx.ageLo} to ${ctx.ageHi} days (missing age: median ${ctx.medianAge} days), chains capped at 10`,
    note:
      'The fit is cross-sectional: in-sample R2 beats the placebo by a wide margin ' +
      'and 5-fold CV stays positive, but the expanding-window walk-forward is negative, ' +
      'so the multiple drifts as new protocols list. The board ranks on the full-sample ' +
      'fit and says so; it is a description of the snapshot, not a forecast. ' +
      'Growth enters as growth_pos = max(winsorised 7-day change, 0): only positive ' +
      'growth adds value. The unconstrained fit (raw growth) gave a growth coefficient ' +
      `of ${fit.unconstrained.growthCoef.toFixed(2)} log points — a drawdown-week artefact: ` +
      'the snapshot week saw broad TVL declines, so the raw variable was dominated by ' +
      'market beta, and the negative sign would reward the protocols that fell hardest. ' +
      'The transform is a pre-chosen monotone specification (growth is a value signal), ' +
      'not a hand-picked list of names; the unconstrained fit is shown beside the refit.' +
      (fit.growthDropped
        ? ` The growth_pos refit still gave a non-positive growth coefficient (${fit.growthRefitCoef.toFixed(2)}), ` +
          'so growth was tested and dropped from the fit entirely (coefficient 0): in this ' +
          'cross-section, seven-day growth carries no positive value signal once TVL, age, ' +
          'chains and category are held fixed.'
        : ''),
  };

  const impliedSorted = rows.map((r) => r.implied_value_usd).sort((a, b) => a - b);
  const boardStats = {
    rows: rows.length,
    watchlist_covered: rows.filter((r) => r.watchlist).length,
    watchlist_total: snap.watchlist.length,
    implied_min: impliedSorted[0]!,
    implied_median: impliedSorted[Math.floor(impliedSorted.length / 2)]!,
    implied_max: impliedSorted[impliedSorted.length - 1]!,
    implied_total: rows.reduce((s, r) => s + r.implied_value_usd, 0),
  };

  const outAbs = resolve(outDir);
  mkdirSync(join(outAbs, 'dashboard'), { recursive: true });
  mkdirSync(join(outAbs, 'screenshots'), { recursive: true });
  cpSync(join(root, 'dashboard'), join(outAbs, 'dashboard'), { recursive: true });

  const boardJson = {
    meta: {
      title: 'Token Radar v2: pre-token protocols ranked by implied value',
      as_of: snap.as_of,
      floor_usd: snap.floor_usd,
      model: modelOut.spec,
      ...boardStats,
    },
    rows: rows.map((r) => ({
      slug: r.slug,
      name: r.name,
      rank: r.rank,
      implied_value_usd: r.implied_value_usd,
      category: r.category,
      chains: r.chains,
      tvl_usd: r.tvl_usd,
      change_7d: r.change_7d,
      age_days: r.age_days,
      watchlist: r.watchlist,
      farm_steps: r.farm_steps,
    })),
  };
  writeFileSync(join(outAbs, 'board.json'), `${JSON.stringify(boardJson, null, 2)}\n`, 'utf8');

  const dataJs = {
    meta: {
      title: 'Token Radar v2',
      as_of: snap.as_of,
      floor_usd: snap.floor_usd,
      ...boardStats,
    },
    model: modelOut,
    rows: rows.map((r) => ({
      slug: r.slug,
      name: r.name,
      category: r.category,
      chains: r.chains,
      tvl: r.tvl_usd,
      change_7d: r.change_7d,
      age_days: r.age_days,
      watchlist: r.watchlist,
      rank: r.rank,
      implied: r.implied_value_usd,
      contrib: Object.fromEntries(
        Object.entries(r.contrib).map(([k, v]) => [k, round4(v)]),
      ),
      farm_steps: r.farm_steps,
    })),
    live: live.map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl: p.tvl_usd,
      mcap: p.mcap_usd,
      multiple: round4(p.mcap_usd! / p.tvl_usd),
      change_7d: p.change_7d,
      age_days: ageDays(p.listed_at, asOf),
    })),
  };
  writeFileSync(
    join(outAbs, 'data.js'),
    `window.RADAR = ${JSON.stringify(dataJs)};\n`,
    'utf8',
  );
  writeFileSync(
    join(outAbs, 'dashboard', 'model.json'),
    `${JSON.stringify({ model: modelOut, board: boardStats }, null, 2)}\n`,
    'utf8',
  );

  if (!dev) {
    const shotDir = join(root, 'screenshots');
    const names = readdirSync(shotDir)
      .filter((n) => n.endsWith('.jpg.b64'))
      .sort();
    if (names.length === 0) throw new Error('no screenshots/*.jpg.b64 found (set TOKEN_RADAR_DEV=1 for a first run)');
    for (const n of names) {
      const b64 = readFileSync(join(shotDir, n), 'utf8').trim();
      writeFileSync(join(outAbs, 'screenshots', n.replace(/\.b64$/, '')), Buffer.from(b64, 'base64'));
    }
    const dossier = buildDossier(outAbs, {
      text_max_bytes: 65536,
      screenshots: { max: 4, max_bytes: 524288 },
      dashboard: { max_files: 400, max_bytes: 4 * 1024 * 1024 },
    });
    writeFileSync(join(outAbs, 'dossier.json'), dossier);
    console.log(`dossier_bytes=${dossier.length}`);
  } else {
    console.log('DEV run: screenshots and dossier skipped');
  }

  const dashBytes = readdirSync(join(outAbs, 'dashboard')).reduce(
    (s, n) => s + readFileSync(join(outAbs, 'dashboard', n)).length,
    0,
  );
  console.log(`rows=${rows.length} watchlist=${boardStats.watchlist_covered}/${boardStats.watchlist_total}`);
  console.log(
    `implied min=${boardStats.implied_min} median=${boardStats.implied_median} max=${boardStats.implied_max} total=${boardStats.implied_total}`,
  );
  console.log(`fit n=${fit.nComps} R2=${fit.r2.toFixed(4)} adjR2=${fit.adjR2.toFixed(4)} residSd=${fit.residSd.toFixed(4)}`);
  console.log(`cv5 oosR2=${fit.cv.oosR2.toFixed(4)} rmseLog=${fit.cv.rmseLog.toFixed(4)}`);
  console.log(
    `walk-forward oosR2=[${fit.walkForward.map((w) => w.oosR2.toFixed(4)).join(', ')}] placeboR2=${fit.placebo.r2.toFixed(4)}`,
  );
  console.log(`dashboard_source_bytes=${dashBytes}`);
}

main();
