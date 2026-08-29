# Table 4 - Learned gate values vs their configured starting points

The evidence for audit finding **Y**. Two things to read here:

1. **λ saturates at its ceiling on every dataset** -- it is a hyperparameter, not a
   learned per-user gate. The earlier reading that it 'sits at its 0.4998
   initialization' was an artifact of Yahoo's init happening to equal its ceiling.
2. **α and θ retain value only where PPD v6 wins.** This is confounded with the
   starting value (Gowalla starts α four times higher); `run_phase3_gate_transfer.py`
   swaps the knobs across datasets to separate the two.

| Dataset | α start | α final | % kept | θ start | θ final | % kept | λ ceiling | λ final | % of ceiling |
|---|---|---|---|---|---|---|---|---|---|
| Yahoo | 0.10 | 0.0003 | 0.3% | 0.75 | 0.0007 | 0.1% | 0.50 | 0.4998 | **100.0%** |
| KuaiRec | 0.35 | 0.0168 | 4.8% | 0.50 | 0.0488 | 9.8% | 0.35 | 0.3500 | **100.0%** |
| Coat | 0.10 | 0.0001 | 0.1% | 0.60 | 0.0216 | 3.6% | 0.30 | 0.2975 | **99.2%** |
