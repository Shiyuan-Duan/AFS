# Final RR-Feature Interpretation and Decision Thresholds

This shallow DecisionTreeClassifier (depth ≤ 5, random_state=42) uses fold-specific RR features calibrated only on training folds.

Key RR features and interpretation:
- sdnn (s): Global variability of RR; higher values indicate AF-related dispersion.
- rmssd (s): Short-term beat-to-beat variability; elevated in AF.
- pnn50: Fraction of |ΔRR| > 50 ms; larger values support AF.
- irreg_index: rr_mad/median_rr; scale-invariant irregularity (higher in AF).
- acf1: Lag-1 autocorrelation of RR; lower periodicity suggests AF.

Typical decision tendencies observed across folds:
- irreg_index > ~0.05–0.08 → AF likely.
- sdnn > ~0.06–0.10 s or rmssd > ~0.05–0.07 s → AF reinforced.
- Lower acf1 strengthens AF; higher acf1 supports Normal.

Determinism and leakage prevention:
- RR peak prominence is set per fold to median(0.5 × P75(envelope)) from training folds only.
- Features are computed per record with no label usage; RR-derived stats are recomputed per fold.

For exact per-fold rule trees, see outputs/feature_catalog.json → rules_per_fold.