# Methodology

## Research question

Can a regime-aware, directed information-diffusion model identify leading venues more effectively than a static lead–lag rule after fees, spread, and slippage?

## Event representation

Each normalized event has a timestamp, venue, instrument, side, volume, mid price, spread, and order-flow imbalance. The built-in demo generator produces timestamped trade events across four BTC/ETH venue-instrument nodes. It makes no external calls, so the repository runs immediately and its outputs are reproducible by seed.

## Diffusion estimator

For event counts \(N_i(t)\), the model computes an exponentially decayed history:

\[
H_i(t) = \sum_{s < t} \gamma^{t-s-1} N_i(s)
\]

For every target node \(j\), it fits a regularized conditional-intensity model:

\[
\lambda_j(t) = b_j + \sum_i a_{ij}H_i(t)
\]

The non-negative \(a_{ij}\) values form the directed diffusion graph. An edge from \(i\) to \(j\) means that source events at \(i\) raise the estimated near-term event intensity at \(j\), conditional on the recent event history supplied to the model.

## Leakage controls

- The Hawkes-style network is fit only on the initial 65% of each replay.
- Event histories use values strictly before the current bin.
- The regime detector uses causal rolling features only.
- Backtest decisions are made before the target holding period begins.

## Economic evaluation

The out-of-sample backtest trades only the three strongest learned cross-venue edges. It uses preceding source order flow to select a target direction, rejects dislocated conditions, and deducts a round-trip taker fee, spread, and state-dependent slippage. It reports net P&L, win rate, a per-trade Sharpe-like score, and maximum drawdown, alongside a baseline that does not avoid dislocation.

## Extension to real data

Replace the replay tensors in `app/data.py` with a loader that emits the same binned fields from timestamp-sorted venue data. Preserve a canonical UTC timestamp, original venue sequence number where available, and a fixed normalization scheme. Walk-forward refits, clock-synchronization checks, cancellation events, queue position, and market-impact calibration are required before interpreting results beyond research experimentation.
