===== MEASURED SPREADS, NOT ASSUMED ONES =====
Method: MT5 records the broker's own spread on every bar. Pulled 75,000 M5
bars per broker straight from each terminal (copy_rates_from_pos), converted
broker points to index points with symbol_info.point, and bucketed by time of
day in Europe/Berlin. This is the broker's own record of what it quoted, not a
published figure and not a single weekend snapshot.

Period: 2025-07-28 .. 2026-08-28 (FundingPips), 2025-07-25 .. 2026-08-28 (FTMO)
All values in INDEX POINTS.

--- FundingPips, GER40 (point 0.01) ---
  window                    n      mean  median   p90    p99     max
  whole 24h             75,000     3.91    2.00   8.50  14.00   22.50
  Xetra 09:00-17:30     29,911     1.97    2.00   2.90   3.60   16.00
  OR bar 09:00             277     1.93    2.00   2.90   3.50    4.00
  entries 09:05-11:00    6,371     1.92    2.00   2.90   3.50    4.00
  US open 15:30-16:00    1,662     1.96    2.00   2.90   3.60   10.90

--- FTMO-Demo, GER40.cash (point 0.01) ---
  window                    n      mean  median   p90    p99     max
  whole 24h             75,000     2.12    1.49   3.55   3.79    7.39
  Xetra 09:00-17:30     29,993     1.26    1.19   1.39   2.39    7.39
  OR bar 09:00             277     1.25    1.23   1.39   1.49    1.59
  entries 09:05-11:00    6,370     1.25    1.23   1.39   1.49    1.69
  US open 15:30-16:00    1,668     1.26    1.23   1.39   1.49    1.49

--- WHAT THIS SETTLES ---
1. The 2.0 used so far for FundingPips was right by accident: the measured
   mean over the entry window is 1.92. config/config.yaml now says 1.92.
2. The 2.0 carried over to FTMO was wrong by 60%. Its real entry-window mean
   is 1.25 - FundingPips quotes 54% wider during the hours we trade.
   config/config.ftmo.yaml now says 1.25.
3. Never use a 24h average: 3.91 vs 1.92 on FundingPips. The overnight hours
   drag it to double the number that applies to this strategy, which only ever
   enters between 09:05 and 11:00.

Cross-check against public sources: no broker publishes a figure for these
symbols. The one corroboration found is trader reporting that FundingPips'
GER40 spread rose from 110 to 290 broker points - 1.10 to 2.90 index points -
which matches the measured median of 2.00 and p90 of 2.90 exactly.

--- EFFECT OF THE CORRECTION ---
ORB, fixed 1R, full year:
  FundingPips  2.00 -> 1.92:  PF 0.84 -> 0.84   -0.086 -> -0.085R
  FTMO         2.00 -> 1.25:  PF 0.87 -> 0.87   -0.069 -> -0.066R
Negligible, and it has to be: 0.75 points against a 75-point R is 1%.

Retest model on M1, where 1R is 27 points and the same 0.75 points is 2.8%:
  broker        maxPB    n      WR     PF     expR
  FundingPips       3   22   50.0%   0.92   -0.042
  FundingPips       6   38   50.0%   0.93   -0.040
  FTMO              3   24   58.3%   1.29   +0.130
  FTMO              6   41   53.7%   1.08   +0.039
  (both  10   ~54 trades   PF 0.72-0.75   about -0.15R)

min_break_or_frac at 1.5R, measured spreads - the divergence between brokers
survives the correction, so it was never a cost artefact:
  frac   FundingPips PF / expR      FTMO PF / expR
  0.00      0.85 / -0.092            0.84 / -0.097
  0.20      1.17 / +0.088            0.95 / -0.025
  0.30      1.27 / +0.131            1.05 / +0.029
  0.40      1.40 / +0.185            1.06 / +0.030

--- STILL ASSUMED ---
slippage_points 1.0 is NOT measured and cannot be taken from bar history. It
stands for the gap between the bar close a signal is based on and the actual
fill on the next bar's open. Only live or demo execution logs can settle it.
==============================================
