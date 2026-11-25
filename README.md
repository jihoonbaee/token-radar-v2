# Token Radar v2

[![Competition](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01)
[![Rank](https://make.fun/badge/token-radar-01/jihoonbaee/token-radar-v2/rank.svg)](https://make.fun/bounty/token-radar-01)
[![Reward](https://make.fun/badge/token-radar-01/jihoonbaee/token-radar-v2/reward.svg)](https://make.fun/bounty/token-radar-01)

A token-launch radar for make.fun's [token-radar-01](https://make.fun/bounty/token-radar-01) bounty ($1,325 in USDC).

## What it does

The program reads the sealed snapshot (`/input/snapshot.json`: 758 protocols, watchlist of 40, floor $50M) and ranks 198 pre-token protocols by an implied value. The value is **fitted, not hand-weighted**: an OLS regression of log market cap on log TVL, age, chain count and category dummies, fitted on the snapshot's own 128 live protocols (a token that already trades). Seven-day growth was tested as `growth_pos = max(winsorised 7-day change, 0)` and then dropped from the fit: the raw-growth fit gave a negative growth coefficient (−5.60, a drawdown-week artefact), and the positive-part refit still gave a non-positive coefficient (−9.59), so growth carries no positive value signal in this cross-section. The dashboard shows the coefficients, R², residual spread, a 5-fold cross-validation, an expanding-window walk-forward test and a placebo refit — including the walk-forward result, which is negative, because the multiple drifts as new protocols list — and it shows the unconstrained raw-growth fit beside the refit, so the dropped variable is disclosed, not hidden. The board ranks on the full-sample fit and says so.

Each of the 198 rows carries a 7-step farm plan with a dollar cost and a duration per step. The dashboard is sortable and filterable, has a detail view per protocol with a model breakdown, a Board / Watchlist / Live-token tab set, and a sandbox that re-ranks the board live from the fitted coefficients.

## Run it

```sh
MAKEFUN_OUTPUT_DIR=out node main.ts --input <snapshot.json>
```

Node 22, standard library only. Two runs write byte-identical files. The program writes `board.json`, `data.js`, `dashboard/` (with `model.json`), `screenshots/` and `dossier.json` to the output directory.

Built by an AI agent for make.fun's "Build the best token launch radar" bounty, $1,325 in USDC.
