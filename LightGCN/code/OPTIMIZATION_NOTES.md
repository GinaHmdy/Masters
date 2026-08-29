# Deferred performance work

> **Update 2026-08-28 — the dominant cost was elsewhere and is now fixed.** `getUsersRating`
> re-ran the full graph propagation once per *user batch* (299 times per pass on Gowalla).
> Caching it in eval mode cut an evaluation pass from 44.6 s to 1.3 s, verified bit-identical.
> See `THESIS_AUDIT.md` finding **AD**. That changes the arithmetic on item 1 below: each
> validation pass now costs ~1.3 s, so merging the two passes would save ~130 s per 100-epoch
> run (~0.04 h) rather than the large win originally hoped for. **Item 1 is no longer worth the
> risk of touching `main.py`.** Item 2 is likewise now negligible.

Not applied yet. Both items are safe-by-construction but must not land while an experiment is
in flight — editing `main.py` mid-ablation is what forced the 2026-08-27 restart (finding **X**),
because arms trained before and after the edit would have used different selection rules.

## 1. Gowalla validation does two full ranking passes per epoch

`main.py`'s v6 training loop calls, every epoch:

- `Procedure.Valid(...)` — ranks all validation users for recall/NDCG/precision
- `Procedure.calculate_average_popularity(..., split="valid")` — ranks them **again** for AvgPop

Since the finding-**X** split fix these two cover the *identical* population (Gowalla's
validation and test users are the same 29,858 people, and Gowalla has no candidate mask), so one
forward pass could feed both, bit-identically.

**Profile before implementing.** Measure validation's share of the 3.5 h/run first — if
validation is 5% of wall time this is a rounding error and the change is pure risk. Suggested
measurement, run in a gap between stages:

```bash
# time one Valid() + one AvgPop call on a trained Gowalla checkpoint
python - <<'PY'
import time, world, Procedure, utils, register
# ... load dataset + checkpoint, then:
# t=time.time(); Procedure.Valid(...); print('valid', time.time()-t)
# t=time.time(); Procedure.calculate_average_popularity(..., split="valid"); print('avgpop', time.time()-t)
PY
```

Then compare against the per-epoch training time to get the share.

**If implemented:** have `Valid()` optionally return the top-K matrix it already computes and
pass it to `calculate_average_popularity`, rather than recomputing scores. Verify by asserting
the AvgPop value is unchanged to full float precision on a fixed checkpoint.

## 2. Evaluation batch size

Peak GPU memory during Gowalla runs was ~2.7 GB of 6.44 GB, so `test_u_batch_size` /
`valid_u_batch_size` can be raised. This changes only how users are chunked into batches and
cannot alter results — but confirm on one checkpoint that metrics are unchanged before adopting
it, since Gowalla retraining is not bit-reproducible (finding **AB**) and a coincidental match
would be misleading. Evaluation from fixed weights *is* deterministic, so an exact match is the
right acceptance test here.
