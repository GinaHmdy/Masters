# Table 5 - Gate transfer: adaptation vs initialization

Finding **Y** observed that alpha and theta retain value on Gowalla and collapse
elsewhere -- but Gowalla also *starts* alpha four times higher, so retention and
initialization were confounded. These arms swap `astd_alpha_base` / `astd_topo_gate_init`
across datasets, holding everything else at the dataset's tuned configuration.

Read the **% kept** columns, not the absolute values: if the collapse tracked the
starting value, a larger start would retain proportionally more.

## Yahoo

*Native configuration: alpha_base=0.1, topo_gate_init=0.75*

| Arm | alpha start | alpha final | % kept | theta start | theta final | % kept | lambda | Recall@20 |
|---|---|---|---|---|---|---|---|---|
| native | 0.10 | 0.0003 | 0.3% | 0.75 | 0.0007 | 0.1% | 0.4998 | 0.138287 |
| gowalla_init | 0.40 | 0.0019 | 0.5% | 0.40 | 0.0007 | 0.2% | 0.4998 | 0.138427 |
| yahoo_init *(= native)* | 0.10 | 0.0003 | 0.3% | 0.75 | 0.0007 | 0.1% | 0.4998 | 0.138287 |

> Arms marked *(= X)* are the **same configuration** as arm X on this dataset -- the override happens to equal the dataset's own value -- so they reuse its result and are **not independent replicates**. See `THESIS_AUDIT.md` finding **AC**.

## Coat

*Native configuration: alpha_base=0.1, topo_gate_init=0.6*

| Arm | alpha start | alpha final | % kept | theta start | theta final | % kept | lambda | Recall@20 |
|---|---|---|---|---|---|---|---|---|
| native | 0.10 | 0.0001 | 0.1% | 0.60 | 0.0216 | 3.6% | 0.2975 | 0.114685 |
| gowalla_init | 0.40 | 0.0004 | 0.1% | 0.40 | 0.0216 | 5.4% | 0.2975 | 0.114685 |
| yahoo_init | 0.10 | 0.0001 | 0.1% | 0.75 | 0.0216 | 2.9% | 0.2975 | 0.114685 |

