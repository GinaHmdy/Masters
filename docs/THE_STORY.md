# The Whole Story
### Making graph recommenders fairer without making them worse — what we tried, what worked, and what didn't

*Written for a computer science student who has never worked on recommender systems.
No prior knowledge assumed. Read it top to bottom.*

---

## Changelog

One line per step, newest last. This is the record of *what we decided and why*, so the sequence
can be followed without reconstructing it from git.

| Date | Step | What happened |
|---|---|---|
| 2026-08-24 | Audit | Independent audit of the contribution's integrity (`THESIS_AUDIT.md`) and a re-derivation of every number from `results/*.json` (`RESULTS_CANONICAL.md`). Established that the Gowalla result is real and the dense-dataset claims were not. |
| 2026-08-24 | Commit `3e3d60c` | PPD v6 / ASTD-PD committed. It had existed only as uncommitted working-tree state — the central contribution was untracked. |
| 2026-08-24 | Step A | **Attribution corrected.** Confirmed from the literature that PPD v1-v4 is a *published method* (arXiv:2510.12959), DAP is published (Chen et al. 2024), DCD is DICE (WWW 2021), and TSP is inherited. README, contribution guide and this document updated. See Part 3. |
| 2026-08-24 | Step A | Framing adopted: **popularity is not always bias** (Part 2.5). Benign popularity = item quality; harmful popularity = exposure amplification. |
| 2026-08-24 | Step B | **Measured** the benign/harmful split using the random-exposure splits, rather than asserting it (`measure_popularity_composition.py`, Part 2.6). Earned share: KuaiRec 26.9%, Yahoo 27.8%, Coat 3.7%, Gowalla unmeasurable. Predictions confirmed on KuaiRec and Yahoo; **the Coat prediction was wrong** and has been corrected in every document rather than quietly dropped. |
| 2026-08-24 | Step F | Added `trivial_baselines.py` (MostPopular + Random on all four datasets). MostPopular reproduces the audit's values exactly. **Random has the lowest AvgPop@20 on every dataset** (15.75 on Gowalla vs the best model's 141.11) — so AvgPop@20 cannot be optimised in isolation (Part 2.8). |
| 2026-08-26 | Phase 1 | **Re-baselined every method on every dataset** under the frozen protocol (3 seeds, mean +/- std, per-user dumps, paired significance tests). Two headline fairness claims died and both had flattered the proposed method; Gowalla's survived unchanged. See Part 5.5. |
| 2026-08-28 | Engineering | **Found that evaluation re-propagated the whole graph once per user batch** — 299 identical propagations per pass on Gowalla, twice an epoch, for 100 epochs. Caching it in eval mode is bit-identical (verified by exact float equality on all 8 metrics) and cuts a Gowalla run from ~3.5 h to ~1.1 h. No number changes; the remaining Phase 2 work drops from ~56 h to ~18 h. Audit finding **AD**. |
| 2026-08-27 | Sanity floors | **Made the plan's two reference rankers an executable gate** (`check_sanity_floors.py`). Every model clears Random everywhere. But on KuaiRec, **every method except PPD v6 loses to MostPopular on NDCG@20** — and v6 beats it only by running at **82.9% of MostPopular's own average popularity**. The sharpest available answer to "when is popularity bias actually bias?". Audit finding **Z**. |
| 2026-08-27 | Verification | **The leak turned out to be inert.** Re-running every affected experiment under the corrected protocol reproduced Yahoo and Coat **bit-identically** and moved Gowalla/KuaiRec by at most 0.35% on one metric — inside the seed noise. Worth reporting precisely because the re-run was actually done: "we found it, fixed it, re-ran, nothing moved" is a claim only available to someone who looked. |
| 2026-08-27 | Phase 2 | **Finding D corrected twice.** λ does not collapse to its *initialization* — it saturates at its configured *ceiling* on every dataset (Gowalla 0.3500/0.35, Yahoo 0.4998/0.50, Coat 0.2975/0.30), so the "personalized fusion gate" is a hyperparameter in costume. And α/θ are **not** universally inert: on Gowalla, the one dataset where v6 wins, they retain 21% and 57% of their starting values while collapsing to zero everywhere else. Audit finding **Y**. |
| 2026-08-26 | Self-audit | **Caught a leak I had just introduced.** Phase 0's scope fix restricted AvgPop to test users — correct for reporting, but `main.py` reuses that same function to build the validation hypervolume that picks checkpoints, so v6 selection was partly scored on the test population. Gowalla is provably immune (valid users == test users); Yahoo had Jaccard 0.18. Fixed, verified bit-identical on reporting, and **Phases 1-2 relaunched** rather than patched. Audit finding **X**. |
| 2026-08-26 | Phase 4 | **Hashed the split files** instead of trusting the directory names. `data/Coat/` and `data/KuaiRec/` are byte-identical to their `unbiased_valid/` variants across train, valid *and* test — so every Coat and KuaiRec result in this thesis was **already** an evaluation on a randomized-exposure test set. The thesis has been entitled to that claim for months and never made it. Audit finding **W**. |
| 2026-08-26 | Phase 2 | **Mechanism-isolation ablation** started (5 arms). Coat complete: the fusion gate `lambda` sits at 0.2973 +/- 0.0001 across every arm *including the ones that delete the streams it fuses*. Gate collapse confirmed as architecture-invariant, not just hyperparameter-invariant. Audit finding **U**. |
| 2026-08-24 | Step B | **Negative result:** per-item quality cannot be estimated from the training log. Structural proxies are collinear with degree (rho 0.98); model affinity `aff_i` predicts quality no better than degree while correlating with it at rho 0.95. The residual decomposition degenerates to raw popularity, so the planned v7 mechanism was abandoned before it was written (Part 2.7). |

---

## Part 1 — The problem

### 1.1 What a recommender system actually is

Strip away the marketing and a recommender system does one thing: **given a user, put every
item in the catalogue in order, best guess first.** Netflix orders films, Spotify orders
songs, a shop orders products. We then show the top 20.

The only raw material we have is a log of what people did:

```
user 7   watched   items 12, 88, 341
user 8   watched   items 88, 90
user 9   watched   items 12, 341, 902
```

That's it. No descriptions, no genres, no ratings — just "who touched what". Learning from
this alone is called **collaborative filtering**, and the intuition is simple: *user 7 and
user 9 overlap a lot, so what one liked the other probably likes.*

### 1.2 The data is a graph

Draw users on the left, items on the right, and an edge whenever someone interacted with
something. That's a **bipartite graph** — two kinds of node, edges only ever going between
the kinds, never within.

```
  USERS            ITEMS
  ┌───┐
  │ 7 │──────────┐ ┌────┐
  └───┘          ├─│ 88 │   ← popular: lots of edges
  ┌───┐          │ └────┘
  │ 8 │──────────┘ ┌────┐
  └───┘            │ 12 │
  ┌───┐         ┌──└────┘
  │ 9 │─────────┘  ┌─────┐
  └───┘            │ 902 │  ← niche: one edge
                   └─────┘
```

Once it's a graph, we can use graph neural networks. That's what **LightGCN** is.

### 1.3 LightGCN in one idea

Give every user and every item a vector — say 64 numbers — called an **embedding**. At the
start they're random. Training nudges them so that *a user's vector points in a similar
direction to the vectors of items they liked.* To recommend, take the dot product of the
user vector with every item vector and sort.

LightGCN's contribution is how those vectors get refined. One round of **message passing**:

> **Replace every node's vector with the average of its neighbours' vectors.**

A user becomes the average of the items they touched. An item becomes the average of the
users who touched it. Do this 2 or 3 times ("layers") and information travels 2 or 3 hops:
you become influenced by items liked by people who liked the same things as you.

There's one refinement. A plain average would let a node with 10,000 neighbours shout down a
node with 3. So each edge is scaled down by the degree of both endpoints:

```
                     1
weight(u,i)  =  ─────────────
                √(deg(u)·deg(i))
```

Remember this formula. **It reduces the popularity problem. It does not remove it.** That gap
is the entire subject of this thesis.

---

## Part 2 — Why popularity bias happens

### 2.1 Real catalogues are wildly unequal

Interaction counts follow a power law. A handful of blockbusters have tens of thousands of
interactions; tens of thousands of niche items have two or three. Draw it and you get a
cliff, then a very long flat **tail**.

### 2.2 "Hub dragging"

Now run message passing on that graph and watch what happens to a niche item.

A niche item has, say, 3 users. Each of those 3 users also touched the mega-popular item.
When the niche item averages its neighbours, it averages 3 users who are *themselves* heavily
influenced by the popular item. So the niche item's vector drifts toward the popular item's
vector.

The popular item, meanwhile, is a neighbour of thousands of users, so it sits near the
*centre* of the whole embedding space. And the centre is exactly where a dot product with an
arbitrary user is large.

Three consequences:

1. **Hub overshadowing** — the popular item is inside almost everyone's neighbourhood.
2. **Representation collapse** — tail items get pulled toward the popular items they happen
   to co-occur with, so they stop being distinguishable from each other.
3. **Echo chamber** — the same head items get recommended to everyone, the recommendations
   generate new interactions on those same items, and the next model trains on an even more
   skewed log.

### 2.3 How we measure the damage

**Average Popularity@20** — for each recommendation we make, look up how many training
interactions that item had, and average over all 20 slots and all users. High number = we
keep recommending blockbusters. Low number = we're reaching into the tail.

Against that we track accuracy:

- **Recall@20** — of the items the user actually went on to like, what fraction did we catch?
- **Precision@20** — of our 20 slots, what fraction were hits?
- **NDCG@20** — like precision, but rewards putting the hits near the top of the list.

**The goal of this thesis, stated exactly:** beat LightGCN and the previous debiasing methods
on *all four at once* — higher Recall, higher NDCG, higher Precision, **and** lower Average
Popularity — on both sparse and dense datasets.

That last part is what makes it hard. Which brings us to the trade-off.

### 2.4 The trade-off, and why the obvious fix fails

The obvious fix: subtract a penalty from popular items' scores.

```
score(u, i)  =  u · i  −  β · popularity(i)
```

It reliably lowers Average Popularity. It also reliably lowers accuracy, because **popular
items are popular partly because they are genuinely good.** A blanket penalty punishes the
blockbuster the user actually wanted just as hard as the one they didn't.

So you get a curve, not a point. Push β up, popularity falls and accuracy falls with it.
Every method is a point on that curve, and for years the field treated moving *up and left* —
better on both axes at once — as impossible.

**Beating that trade-off is the thesis's target.**

### 2.5 The key idea: popularity is not always bias

Read §2.4 again and notice the phrase doing all the work: *popular items are popular partly
because they are genuinely good.* That word **partly** is the whole thesis.

Popularity has two sources, and they are completely different things:

- **Benign popularity — it's actually good.** A million people watched it because a million
  people genuinely wanted to. Here, popularity *is* signal. Removing it is vandalism.
- **Harmful popularity — it was just shown more.** It appeared on the front page, so it got
  clicked, so the algorithm showed it more, so it got clicked more. Nothing about the item
  earned this. This is the feedback loop, and this is what a debiaser should remove.

```
        observed popularity  =  benign (quality)  +  harmful (exposure amplification)
                                 ^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 keep this            remove this
```

This is not our invention — it is a known result. Zhao et al., *Popularity Bias Is Not Always
Evil: Disentangling Benign and Harmful Bias for Recommendation* (IEEE TKDE 2022,
[arXiv:2109.07946](https://arxiv.org/abs/2109.07946)) separates item *quality* from *conformity*.
PDA (Zhang et al., SIGIR 2021) goes further and deliberately *uses* popularity instead of
deleting it.

**Now here is why it explains everything in Part 5.** Every method in this repository — PPD
v1-v4, DAP, and PPD v6 — removes popularity *uniformly*. It does not know which kind it is
removing. And:

- removing **benign** popularity costs accuracy roughly one-for-one — you are deleting real
  information;
- removing **harmful** popularity is almost free — you are deleting an artifact.

A method that can't tell them apart can only **slide along** the curve from §2.4. A method that
*can* tell them apart **moves the curve**. That is the difference between what we built and what
the goal actually requires — and it is why the honest answer to "why didn't v6 win everywhere?"
is not "it needs more tuning".

### 2.6 Don't assume the split — measure it

Here is where it would be easy to write a nice story and move on. We didn't, and the measurement
changed the story.

**How you can actually measure it.** Three of the four datasets ship a special test set collected
under **random exposure**: Coat and Yahoo!R3 asked users to rate items *chosen for them at
random*, and KuaiRec's test matrix is *fully observed* — every user, every item. That removes the
exposure confound completely. If an item is shown to a random sample of people, the fraction who
like it **is** its quality.

So: regress training popularity on that measured quality. Whatever the fit explains is earned;
the rest is amplification. `LightGCN/code/measure_popularity_composition.py` does exactly this.

| Dataset | How much of popularity is *earned* | correlation (quality, degree) | Verdict |
|---|---|---|---|
| **KuaiRec** | **26.9%** | **+0.74** | popularity substantially earned |
| **Yahoo** | **27.8%** | **+0.43** | popularity substantially earned |
| **Coat** | **3.7%** | **+0.17** | popularity almost entirely amplified |
| Gowalla | *no random-exposure split exists* | — | unmeasurable — so we don't guess |

**Now the predictions, and the honest scorecard.** Where popularity is earned, removing it should
cost accuracy. Where it is amplified, removing it should help.

| Dataset | Predicted | What actually happened | |
|---|---|---|---|
| KuaiRec | debiasing costs accuracy | every model — ours included — loses to plain most-popular | ✅ **confirmed** |
| Yahoo | debiasing costs accuracy | v6 loses Recall, NDCG and Precision to LightGCN | ✅ **confirmed** |
| Coat | debiasing has room to help | v6 loses *every* accuracy metric | ❌ **wrong** |
| Gowalla | unmeasurable | v6 wins 3/4 | — untestable |

**Two out of three, and the miss is the useful part.** The two results that looked like the
method failing — KuaiRec and Yahoo — turn out to be explained by a property of the *data*,
measured without reference to any model. That is a much stronger thing to say in a viva than "we
need more tuning."

Coat is genuinely not explained. Its popularity is 97% amplification, so debiasing *should* have
helped, and it didn't. So we say so, and we look elsewhere: Coat has 281 test users and 300
items, and v6 carries four embedding tables where LightGCN carries two. That smells like a model
too big for the dataset, not a popularity effect. It stays on the open-questions list rather than
being quietly folded into a story that happens to be convenient.

> **This is the part to copy.** A framework that explains *everything* usually explains nothing —
> it has just been fitted to the results after the fact. One that makes four predictions, gets
> three right and names the fourth as unexplained is doing real work. Write down the prediction
> **before** you look.

### 2.7 The thing we tried to build, and couldn't

The obvious next move: if popularity splits into earned and amplified, estimate that split *per
item* and remove only the amplified part. That was the plan. It does not work, and the reason is
worth more than the mechanism would have been.

We tried three ways to estimate item quality **from the training log alone**:

| Attempt | Idea | Result |
|---|---|---|
| Structural proxies | attention share (`1/deg_u` of consumers), explorer mix, 2nd-order density | The density term correlates with degree at **0.98** — it is degree wearing a hat. The apparent R² of 0.81–0.99 was degree explaining degree. On KuaiRec the "most harmful" items came out with degree **1**. Nonsense. |
| Model affinity | `aff_i = ê_i · mean_u(ê_u)` — how much the trained model's users like item *i* | Predicts true quality **no better than raw degree** (+0.70 vs +0.71 on KuaiRec), while correlating with degree at **0.95**. The embedding had simply *learned* popularity — which is a neat demonstration of the very bias we're removing, and useless as a fix. |

In every case the "harmful residual" ended up correlated with degree at +0.89 to +0.98 — meaning
the whole decomposition collapsed back into plain popularity, doing nothing.

**Why it was always going to fail.** The training log records what people did *given what they
were shown*. It contains no counterfactual — nothing about what they would have done had they
been shown something else. You cannot recover exposure-free quality from exposure-contaminated
data by rearranging it. That is precisely why Coat, Yahoo!R3 and KuaiRec were *built* with
random-exposure splits, and why TIDE reaches for timestamps instead.

So the split becomes a **diagnostic that tells you how hard to debias each dataset**, rather than
a per-item correction inside the loss. Less glamorous. Actually usable.

> **Lesson.** We had a gate written down in advance: *"if the residual is just a monotone function
> of degree, the decomposition failed."* It was. Because the gate existed, this cost an hour
> instead of a month. Write the falsification test before you write the feature.

### 2.8 The scoreboard has a hole in it, and a random recommender finds it

Run the dumbest possible baselines — recommend the 20 most popular items to everyone, and
recommend 20 items at random — and score them on the same four metrics
(`LightGCN/code/trivial_baselines.py`):

| Dataset | Method | Recall@20 | NDCG@20 | Precision@20 | AvgPop@20 |
|---|---|---|---|---|---|
| Gowalla | MostPopular | 0.040822 | 0.031229 | 0.013635 | 683.99 |
| Gowalla | **Random** | 0.000486 | 0.000327 | 0.000174 | **15.75** |
| KuaiRec | MostPopular | **0.123342** | **0.313162** | **0.266525** | 1948.98 |
| KuaiRec | **Random** | 0.001838 | 0.009792 | 0.009764 | **86.86** |
| Coat | MostPopular | 0.095386 | 0.051185 | 0.019039 | 38.37 |
| Coat | **Random** | 0.055013 | 0.029549 | 0.013523 | **11.68** |

Look at the AvgPop@20 column. **Random wins it on every single dataset, by a mile.** 15.75 on
Gowalla, where the best debiaser in this thesis manages 141.11 and the previous best prior-art
method manages 139.86.

Of course it does — recommending noise recommends unpopular things. But it means:

> **Average Popularity@20 is a metric that a random number generator wins outright.**

So an "85% popularity reduction" is not, by itself, an achievement. It is only meaningful *held
against accuracy*, which is exactly why the goal names all four metrics together and why the
selection criterion is a 4-dimensional Pareto hypervolume rather than popularity alone. If you
ever find yourself proud of an AvgPop number, check where Random sits first.

And the MostPopular row is the other half of the lesson: on KuaiRec it beats every model in this
thesis by roughly 9x on Recall and 15x on NDCG. That single row reframes an entire chapter — see
Part 5.3.

**One consequence for measurement.** AvgPop@20 punishes a recommender for surfacing a genuinely
excellent popular item, *and* rewards it for recommending noise. It stays, because it is named in
the goal — but on its own it is the wrong scoreboard, and it needs companions: accuracy on the
long tail, catalogue coverage, and — where an unbiased split exists — the *unearned* share of the
popularity being recommended.

---

## Part 3 — What already existed when this work started

This matters, and thesis write-ups often blur it. Three components were **already in the
codebase**, written by a previous author, before this work began:

| Component | What it does | Origin |
|---|---|---|
| **TSP** (Topological Simplex Propagation) | smooths embeddings over a higher-order structure | inherited from the prior author (present at commit `d3cad59`) |
| **PPD v1–v4** (Post-hoc Popularity Debiasing) | removes a "popularity direction" from embeddings | **published**: Islam, Wang & Zheleva, *Post-hoc Popularity Bias Correction in GNN-based Collaborative Filtering*, [arXiv:2510.12959](https://arxiv.org/abs/2510.12959) (Oct 2025) |
| **DAP** (Degree-Aware Propagation) | clusters nodes and subtracts high/low-degree anchors | **published**: Chen et al. 2024 — and itself a baseline *in* the PPD paper |
| **DCD** (`run_dcd_sweep.py`) | splits interest from conformity into two embeddings | **published**: DICE, Zheng et al., WWW 2021 |

This was checked properly rather than assumed. "PPD" in this repository is not a house name for
a house method — it is a real published algorithm, and `utilities.project_and_remove` implements
that paper's debiasing step. The paper even uses the same three evaluation datasets (Coat,
KuaiRec, Yahoo! R3). The same group has since published PPD's successor, **DPAA**
([arXiv:2605.11145](https://arxiv.org/abs/2605.11145), May 2026).

**Why finding this out is good news, not bad.** If PPD and DAP were merely earlier code from the
same lab, then "beating PPD" would look uncomfortably like beating your own ablation. Because
they are published methods with published numbers, beating them is ordinary, legitimate baseline
comparison — which is exactly the shape a thesis wants. The one thing that would have been fatal
is claiming to have invented them, which the README did until this was corrected.

So the job was not "invent debiasing". It was: **here are three existing methods — build
something that beats all of them, on all four metrics, on sparse and dense data.** That is a
perfectly good MSc problem, and it's the right way to frame the work.

What *this* work added: PPD v5, PPD v6 / ASTD-PD, the ego-weighting option, several real bug
fixes, and the whole evaluation and experiment infrastructure.

### 3.1 TSP — and a lesson about circular reasoning

The idea sounds sophisticated. Beyond edges (which connect 2 nodes), you can build
**simplices**: a triangle is a 2-simplex connecting 3 nodes, and so on. The mathematics of
these is algebraic topology, and there's a Laplacian operator for smoothing over them.

Here's what the code actually does:

1. Take the current item embeddings.
2. Connect two items if their embeddings have cosine similarity above a threshold.
3. Build simplices on that graph, smooth over them, write the result back.

**Spot the problem.** The graph in step 2 is built from *the model's own embeddings*. So the
"structure" we smooth over is just a restatement of what the model already believes. It's
circular — like proofreading your own essay by rereading it and agreeing with yourself.

And a detail: the setting used in every experiment is `k_tsp = 1`, which means **only edges
were ever built — no triangles, no higher-order simplices at all.** The topology library is
imported and, at that setting, does nothing beyond ordinary graph smoothing.

The measured result is exactly what you'd predict: **TSP changes the numbers by roughly
0.0%** on every dataset.

> **Lesson 1.** If your structure is derived from your model's own output, it cannot teach
> your model anything.

### 3.2 PPD v3 and v4 — a genuinely good idea

This one is more interesting. Instead of penalising scores, find the **direction in embedding
space that means "popular"**, and delete that component from every vector.

For each user, split the items they interacted with into two weighted groups — "they picked
this because it's popular" vs "they picked this because they wanted it" — average each group,
and take the difference:

```
d_pop  =  normalize( centroid_popular  −  φ · centroid_preference )
```

Then remove each embedding's component along `d_pop`. Geometrically: flatten the space along
the popularity axis, leave everything perpendicular to it untouched.

- **v3** does this once, on the starting embeddings.
- **v4** does it at *every* message-passing layer, so popularity can't creep back in.

**Result:** v4 cuts Average Popularity hard — on Gowalla, 214 → 140. But accuracy falls with
it: Recall drops from 0.1405 to 0.1275, about 9%. Classic trade-off. A point that moved down
and left, not up and left.

### 3.3 A real bug worth understanding

Removing a component along a unit direction `d` is:

```
x  ←  x − (x · d) d
```

The code had a scale factor β on that:

```
x  ←  x − β (x · d) d
```

Look at what happens at β = 2:

```
x  ←  x − 2(x · d) d
```

That is a **reflection**, not a removal. The popularity component isn't deleted — it's
flipped to point the *opposite* way, with the same magnitude. The vector's length doesn't
even change. And KuaiRec experiments were being run at exactly β = 2.

This was found and fixed (commits `e8d12d0`, `1fde1f5`). Good catch, real contribution.

But there's a sting in the tail. After the fix, β is no longer used in that function at
all — the argument is passed in and never read. So **`--beta` now has no effect on PPD v3**,
even though per-dataset β values are still swept, tuned, and reported throughout the project.

> **Lesson 2.** After you delete a term, grep for its parameter. Dead knobs keep showing up
> in results tables long after they stopped doing anything.

---

## Part 4 — PPD v6 / ASTD-PD, the new method

The reasoning behind v6 is sound, and worth following.

**The diagnosis:** v3 and v4 fail because they force *one* set of embeddings to be accurate
and debiased simultaneously. Those goals conflict, so you get a compromise that's mediocre at
both.

**The fix:** stop compromising. Keep **two** sets of embeddings.

```
                    interaction graph
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   ACCURACY STREAM                DEBIASED STREAM
   never debiased,                popularity removed
   just get it right              at every layer
            │                             │
            └──────────────┬──────────────┘
                           ▼
                    fuse per user
              score = s_acc + λ(u)·t(i)·s_deb
```

Four ingredients:

**1. Two streams.** One is protected and optimised purely for accuracy. The other is
aggressively debiased. Neither has to compromise.

**2. A learned popularity subspace.** Rather than hand-crafting `d_pop`, sort items into five
popularity buckets, take differences between bucket averages, and run QR decomposition to get
a clean orthonormal basis for "the popularity subspace". Update it slowly with a moving
average. This is a genuine improvement over v3/v4 — and unlike them, it uses **real item
degree** rather than an embedding-derived proxy.

**3. Tail reweighting.** At scoring time the debiased stream is weighted by
`t(i) = (1 − popularity(i))^φ`, so it only speaks up about tail items. During training, tail
positives get extra weight in the loss. **Remember this ingredient — it turns out to be the
important one.**

**4. A personalized gate.** `λ(u)` is a small neural network that decides, per user, how much
to listen to the debiased stream. Someone who only ever watches blockbusters gets a low λ;
someone who explores gets a high one.

That's a well-motivated architecture. So — did it work?

---

## Part 5 — The results, honestly

### 5.1 Gowalla (sparse): yes, and this is the real result

| Method | Recall@20 ↑ | NDCG@20 ↑ | Precision@20 ↑ | AvgPop@20 ↓ |
|---|---|---|---|---|
| LightGCN | 0.140533 | 0.115921 | 0.042134 | 214.05 |
| TSP | 0.140183 | 0.116862 | 0.042374 | 210.26 |
| PPD v3 | 0.140396 | 0.116456 | 0.042227 | 200.17 |
| PPD v4 | 0.127509 | 0.107013 | 0.038061 | **139.86** |
| **PPD v6** | **0.142838** | **0.121338** | **0.043473** | 141.11 |

This is a real win. Better than LightGCN, TSP and PPD v3 on **all four** metrics
simultaneously — that's the up-and-left move the field said was hard. And it gets almost all
of PPD v4's popularity reduction (214 → 141) while being **12% more accurate** than v4.

It misses the clean sweep by one number: AvgPop 141.11 vs v4's 139.86 — a 0.9% gap. One
metric, one dataset, from the full stated goal.

The margins are 3–12× the run-to-run noise, so this isn't luck.

### 5.2 Yahoo (dense): no — and here's the methodology trap

The honest Yahoo result: **PPD v6 loses to LightGCN** on Recall, NDCG and Precision.

But the write-up reports a *win*. Where did it come from? A script called
`refine_yahoo_ppd_v6.py`, whose comment block states the exact target numbers to beat, tries
64 configurations, and **picks the one that scores best on the test set.**

That is the cardinal sin of empirical machine learning. The test set is your only estimate of
performance on unseen data. The moment you choose anything using it — a hyperparameter, a
random seed, which of 64 configs to report — it stops being unseen, and the number it gives
you is optimistic in a way you can no longer measure.

The proof is in the repo itself. The same experiment, selected properly on the *validation*
set, produces a loss on all four metrics. Selected on the test set, it produces a win. Same
code, same data, same model — only the selection rule changed.

> **Lesson 3.** Pick your configuration on validation. Touch the test set once, at the very
> end, and report whatever it says.

And even the test-selected "win" is +0.001115 Recall against a run-to-run standard deviation
of 0.002547. The margin is **smaller than the noise.** Without error bars it looked
meaningful. With them, it isn't.

> **Lesson 4.** A difference without an error bar is not a result.

**Re-measured, 2026-08-26 (Phase 1, frozen protocol, 3 seeds).** With the metric-scope bug fixed
and fairness metrics added, Yahoo turns out to be v6's *cleanest trade* rather than a plain loss:

| Metric | LightGCN | PPD v6 | Verdict |
|---|---|---|---|
| Recall@20 | 0.141245 +/- 0.002547 | 0.138287 +/- 0.000375 | -1.16 sigma, **not significant** (p=0.539) |
| NDCG@20 | 0.064582 | 0.062616 | -2.12 sigma, **not significant** |
| Precision@20 | 0.010995 | 0.010827 | -1.61 sigma, **not significant** |
| AvgPop@20 | 595.72 | **527.13** | **-11.5%** |
| APLT@20 | 0.1101 | **0.1428** | **+29.7%** |
| Coverage@20 | **0.6800** | 0.6253 | -8.0% |

So the accuracy loss is real in direction but too small for three seeds to distinguish from
noise, while the fairness gain is several times the seed spread. This is *not* a Pareto win —
Coverage moves the wrong way, and "not significant" at n=3 is a statement about statistical
power, not a proof of equivalence. But it is the one dataset where "fairness improved at no
*demonstrated* accuracy cost" is defensible exactly as written.

One caveat travels with every Yahoo number in this document: only **16.6%** of Yahoo users
appear in the test split, so all of it describes a minority of the user base.

### 5.3 KuaiRec (dense): the most instructive failure

The write-up's headline is a **32.8× improvement** in Recall over LightGCN.

Whenever you see a number like that, ask one question: *what is the baseline doing?*

The LightGCN baseline on KuaiRec scores Recall@20 = 0.00074. Its Recall@5 is **exactly
zero** — in five attempts across 1,410 users it never once got a single item right. That's
not a baseline.

**And it turned out not to be a broken training run either — it was a broken ruler.** Asking
"what is the baseline doing?" was the right instinct, but the first answer was wrong. The model
was fine; the *evaluation* ranked over all 10,728 items while KuaiRec's test split covers only
3,247, so 99.1% of every top-20 consisted of items that could not be scored at all. Restrict the
candidates to the evaluated universe and the same checkpoint scores **0.155524** — a 210× jump
from a one-line fix, with no retraining. The tell was that LightGCN sat 19× *below random*: a
weak model drifts toward random, it does not fall an order of magnitude beneath it.

So here is the check that should have been run on day one. Ignore personalization entirely.
Rank items by raw popularity. Recommend the same 20 items to everybody:

| Dataset | MostPopular Recall@20 | Best model in the thesis |
|---|---|---|
| Gowalla | 0.040822 | 0.142838 ✓ 3.5× better |
| Yahoo | 0.090639 | 0.142360 ✓ 1.6× better |
| Coat | 0.095386 | 0.125553 ✓ 1.3× better |
| **KuaiRec** | **0.123342** | **0.177384 ✓ 1.4× better** *(was 0.013903 ✗ 9× worse before the eval fix)* |

The MostPopular check was still the right check to run on day one — it just did its job in an
unexpected way. It did not prove the models were worse than a popularity ranker; it proved the
measuring instrument was broken, because *every* model failing that badly at once is a property
of the harness, not of the methods.

The reframing that came out of this still stands, and it survives the correction intact. Look at
KuaiRec's shape: about **103 relevant items per user** out of 10,728. It is extraordinarily dense.
When a user likes 1% of the entire catalogue, popular items really are a strong guess, and
*popularity there is not purely a bias — it carries genuine signal.* MostPopular reaching
Recall@20 = 0.123 on a dataset where it reaches 0.041 on Gowalla is real, and it is not an artifact.

Which reframes the whole thesis question. Not *"how do we remove popularity bias?"* but:

> **"When is popularity bias actually bias, and when is it information?"**

And the corrected KuaiRec numbers answer it more sharply than the broken ones ever could. The old
reading was fatalistic: popularity is the signal here, so debiasing must cost accuracy. That turns
out to be false. **PPD v4 holds LightGCN's accuracy (0.155985 vs 0.155524) while cutting average
recommended popularity by 33% (684 vs 1025).** [RETRACTED — see below.] So even on the densest, most popularity-driven
catalogue in the set, a third of the popularity concentration is *exposure* effect that can be
removed for free — and the rest is the quality signal that cannot. That is the benign/harmful split
the thesis argues for, measured rather than assumed.

> **Lesson 5.** Always run the dumbest possible baseline first. If it beats everything at once,
> suspect your ruler before you believe the result.

### 5.4 Coat (small): a trade, not a rout

PPD v6 loses to every baseline on every accuracy metric by roughly 2 standard deviations.
Coat is quietly missing from the results section of the project write-up, while still being
counted in the phrase "four diverse datasets". If a dataset is in your claim, it's in your
table.

**Re-measured with fairness metrics (2026-08-26), the picture is legible rather than simply
bad.** The accuracy loss is real and significant (Recall -2.07 sigma, NDCG -5.62 sigma), but it
buys a large exposure gain: Coverage@20 **0.3622 -> 0.5533 (+53%)** and APLT@20 **0.2240 ->
0.3117 (+39%)**. That is a stated trade, and a thesis may report a trade. What it may not do is
report only the half that flatters it — which is what omitting Coat from the results section did.

### 5.5 The Phase 1 scoreboard, and what it cost

Re-running everything under one frozen protocol produced the table the thesis should have had
from the start — and it was expensive, because two of the three headline fairness numbers did
not survive it. **Both had flattered the proposed method.**

| Dataset | Accuracy vs LightGCN | Fairness vs LightGCN | Verdict |
|---|---|---|---|
| **Gowalla** | NDCG **+24.5σ** and Precision **+47.5σ** significant on both tests; Recall +12.6σ significant per-user but **p=0.056 across seeds — not significant** | AvgPop **−34%**, Coverage **+258%**, APLT **+257%** | **Pareto win, with the recall caveat stated** |
| **Yahoo** | all three losses **not significant** (-1.2 to -2.1 sigma) | AvgPop -11.5%, APLT +29.7%, Coverage -8.0% | clean trade |
| **Coat** | Recall -2.07 sigma (significant loss) | Coverage +53%, APLT +39% | stated trade |
| **KuaiRec** | Recall **+14%**, NDCG **+30%** (significant) | AvgPop **+41% WORSE**, Coverage **-80%**, APLT **-84%** | **fairness reversal** |

The KuaiRec row is the one to sit with. A method whose entire purpose is popularity debiasing
ends up **41% more popularity-biased than the baseline it debiases**, recommending 3.2% of the
catalogue — roughly 340 items out of 10,728. It bought its accuracy win by collapsing onto the
head. The earlier "-5.6% AvgPop, dominant on KuaiRec" was an artifact of measuring AvgPop over
all users while measuring accuracy over the 19.6% in the test split.

That is a bad result to bury and a strong chapter to lead with, and the reference rankers show
exactly why. On KuaiRec:

| Ranker | NDCG@20 | AvgPop@20 |
|---|---|---|
| **MostPopular** — same 20 items for all 1,410 users | **0.313162** | 1948.98 |
| LightGCN | 0.304830 | 1148.79 |
| PPD v4 | 0.303910 | 1158.82 |
| PPD v3 | 0.119512 | 392.20 |
| **PPD v6** | **0.395928** | **1616.56** |

**Every method in this study except v6 loses to a ranker with no personalization whatsoever.**
And v6 beats it by running at **82.9% of MostPopular's own average popularity** — it wins not by
being less popularity-driven but by being *almost as* popularity-driven while keeping some
personalization.

On KuaiRec popularity is 26.9% *earned* (Part 2.6, measured not assumed) on a catalogue with ~103
relevant items per user. Under those conditions the popularity ranker is close to optimal, and a
debiaser is being asked to throw away the most predictive feature in the data. A gate trained to
maximise a hypervolume containing accuracy correctly learns that riding popularity is the winning
move. **The method did what the data told it to.** The failure is not the optimiser's; it is in
expecting a debiaser to help where popularity is earned.

Note also **PPD v3's row is a degenerate solution, not a favourable trade**: the best AvgPop of
any method (392.20) bought by losing to MostPopular on *all three* accuracy metrics. Fairness
metrics that good with accuracy that bad are a warning sign, and the write-up should label it.

Why believe Gowalla, then, when two of its siblings broke? Precisely because Gowalla was
re-measured *after* both fixes and did not move: its test users are 100% of its users, so the
metric-scope bug could not touch it. The result that survived the audit is the one that never
depended on the audit's mistakes.

---

## Part 6 — The twist: the clever parts weren't doing anything

This is the most important section, and it comes from the project's own logs.

The code records, every epoch, what its three learned gates are doing:

- **θ** — how much of the topological/simplex stream to use
- **α** — how strongly to remove the popularity subspace
- **λ** — the personalized fusion weight

Across **all 108** Yahoo configurations:

| Gate | Controls | Value it converged to |
|---|---|---|
| θ | the simplex topology stream | **0.0007** (started at 0.48) |
| α | the spectral popularity removal | **0.0003** |
| λ | the personalized fusion gate | **0.4998** — which looked like its starting value, but see 6.2: it is its **ceiling** |

Read that carefully:

- θ ≈ 0 means the **topological stream was switched off** by training. Ingredient 1: inert.
- α ≈ 0 means **almost no popularity was being removed** by the spectral projection.
  Ingredient 2, the QR subspace — the mathematical centrepiece: inert.
- λ stuck at 0.4998 looked like the "personalized" gate **never learning anything**.

That was the reading from the logs alone. Phase 2's ablation then tested it directly, and
**two of those three conclusions had to be revised — both times toward something sharper.**

### 6.1 The spectral projection is not "nearly" inert. It is numerically absent.

On Yahoo, arm (b) deletes the spectral projection entirely. Against arm (a), full v6:

| | Recall@20 | NDCG@20 | AvgPop@20 | APLT@20 |
|---|---|---|---|---|
| (a) full v6 | 0.138287 | 0.062616 | 527.13 | 0.1428 |
| (b) no spectral | 0.138287 | 0.062617 | 527.13 | 0.1427 |

Identical to six decimal places on Recall. Removing the mathematical centrepiece of the
architecture changes *nothing measurable*. That is the strongest form this finding can take:
not "the effect is small" but "the effect is not there".

### 6.2 λ was not stuck at its initialization. It saturates at its ceiling.

The "0.4998 = its starting value" reading was a coincidence — Yahoo's `astd_fuse_lambda_max`
happens to equal its init. Across datasets with *different* ceilings:

| Dataset | λ ceiling (`astd_fuse_lambda_max`) | λ final | % of ceiling |
|---|---|---|---|
| Gowalla | 0.35 | 0.3500 | **100.0%** |
| Yahoo | 0.50 | 0.4998 | **99.96%** |
| KuaiRec | 0.35 | 0.3500 | **100.0%** |
| Coat | 0.30 | 0.2975 | **99.2%** |

λ pins itself to whatever ceiling the config file sets, on every dataset, in **every ablation
arm — including the arms that delete the streams it is supposed to be fusing**. A gate that
does not move when its inputs are removed is not a gate. `astd_fuse_lambda_max` does not
*bound* the fusion weight; it **is** the fusion weight. The "personalized Pareto fusion layer"
is a hyperparameter wearing a costume.

### 6.3 And the mechanisms are alive exactly where the method wins

Finding θ ≈ α ≈ 0 on Yahoo was generalized to "all mechanisms are inert everywhere". That
generalization was wrong:

| Dataset | α start → final | θ start → final | v6 outcome |
|---|---|---|---|
| **Gowalla** | 0.40 → **0.083** (21% kept) | 0.40 → **0.227** (57% kept) | **Pareto win** |
| Yahoo | 0.10 → 0.0000 | 0.75 → 0.0004 | loss (n.s.) |
| Coat | 0.10 → 0.0001 | 0.60 → 0.0216 | loss |

On the one dataset where PPD v6 wins, the machinery stays switched on. On the two where it
loses, gradient descent switches it off. Read optimistically, that is the mechanism-level
version of this thesis's whole argument: the debiaser activates in proportion to how much of
the popularity signal is *exposure* rather than *quality*, and Yahoo/KuaiRec have 27.8% / 26.9%
measured earned popularity while Gowalla is sparse.

Read sceptically — and it must be read sceptically — Gowalla also *starts* α four times higher,
so this could be decay from a bigger number rather than adaptation. `run_phase3_gate_transfer.py`
swaps the two knobs across datasets to settle it. **Until that lands, 6.3 is a hypothesis, not
a result.**

### 6.4 So what *does* produce the popularity reduction? Not what I claimed.

The old answer was "the tail reweighting — the simplest thing in the design". The ablation says
otherwise. Arm (e) turns off **all three** debiasing mechanisms, and on Yahoo it still delivers:

| Metric | LightGCN | (e) all debiasing OFF | (a) full v6 |
|---|---|---|---|
| Recall@20 | 0.141245 | 0.141699 | 0.138287 |
| **AvgPop@20** | 595.72 | **529.71 (−11.1%)** | 527.13 (−11.5%) |
| APLT@20 | 0.1101 | 0.1224 (+11.2%) | 0.1428 (+29.7%) |

**Arm (e) reproduces ~96% of v6's entire popularity reduction with every named mechanism
disabled.** The reason is an error in my own ablation design: arm (e) still trains through a
**popularity-stratified negative sampler**, which draws negatives by popularity decile. That is
a debiaser. "No debiasing" was never no debiasing.

So the honest current statement is: *most of the measured popularity reduction on Yahoo comes
from the negative sampler — a component the contribution is not named after and does not
claim.* Arm (f), which forces uniform negative sampling, is queued to isolate it. Until it
runs, **no sentence of the form "mechanism X produces the popularity reduction" is safe**,
because the sampler is inside every arm including the control.

That is a smaller claim than the one I started with, and a much better-evidenced one.

### 6.5 Why this is a better thesis than the one I set out to write

Restated for a viva:

> I built a dual-stream architecture with a spectral popularity subspace, a topological routing
> gate and a personalized fusion gate. I instrumented all three and then ablated them. The
> spectral projection turned out to be numerically absent — deleting it changes results in the
> sixth decimal place. The fusion gate turned out to be a hyperparameter: it saturates at its
> configured ceiling on every dataset and in every arm, including arms that delete its own
> inputs. And when I built a control arm with all three mechanisms disabled, it still reproduced
> most of the popularity reduction — because the training recipe contains a popularity-stratified
> negative sampler I had not counted as a mechanism.

Every one of those sentences is a negative result about my own design, and every one is backed
by a measurement rather than a suspicion. That is worth more than "my complicated model won by
0.8%", which is what the alternative write-up would have said. The requirement is to claim it
**deliberately**, in the thesis, rather than have an examiner find it in the logs.

> **Lesson 6.** Instrument your mechanisms and check they're actually running. A model can
> improve for reasons that have nothing to do with your idea.

> **Lesson 6a.** A control arm is only a control if you can name everything it still does.
> Mine still had a debiaser in it.

---

## Part 7 — The lessons

1. **Structure derived from your own model teaches it nothing.** (TSP)
2. **After deleting a term, hunt down its parameter.** (dead `--beta`)
3. **Choose on validation. Touch test once.** (Yahoo)
4. **A difference without an error bar is not a result.**
5. **Run the dumbest baseline first.** (KuaiRec)
6. **Instrument your mechanisms — verify they're switched on**, then *ablate* them to check
   the instruments. (θ, α, λ: the logs said "nearly zero"; the ablation said "numerically
   absent", "pinned to a hyperparameter", and "alive on Gowalla" — three sharper claims.)
6a. **A control arm is only a control if you can name everything it still does.** The
   "no debiasing" arm still contained a popularity-stratified negative sampler, which accounted
   for ~96% of the popularity reduction it was supposed to be a baseline against.
7. **Say what you inherited — and check whether it was published.** Nothing is lost by
   attributing prior work, and everything is lost by not. In this project the check mattered
   more than expected: the components assumed to be "the lab's earlier code" turned out to be
   *published papers* (PPD = arXiv:2510.12959, DAP = Chen et al. 2024, DCD = DICE, WWW 2021).
   Ten minutes of searching changed the framing of the whole thesis — and changed it in the
   candidate's favour. Do that search **before** writing the contributions section, not after.

---

## Part 8 — Where this should go next

The clean version of this thesis is smaller and much stronger:

> ### Correction, 2026-08-25
>
> Two claims in Part 5 are withdrawn, both killed by protocol bugs rather than by re-tuning,
> and **both had flattered the proposed method**:
>
> 1. *"PPD v4 cuts KuaiRec popularity 33%"* — the AvgPop metric ran over **all** users while
>    accuracy ran over **test** users (16-20% overlap on the dense datasets). On test users
>    PPD v4's KuaiRec AvgPop is **+0.9%, i.e. worse**. It had been suppressing popular items
>    for the ~80% of users nobody was scoring. See audit finding **T**.
> 2. *"TSP is a weak baseline"* — on Coat TSP builds **zero edges** and does not run at all;
>    its numbers are byte-identical to LightGCN. See audit finding **S**.
> 3. *"PPD v6 dominates KuaiRec"* (added 2026-08-26) — same metric-scope bug. Corrected, v6's
>    KuaiRec AvgPop is **+41% worse** than LightGCN's and its Coverage is **-80%**. It is an
>    accuracy win and a fairness reversal. See Part 5.5.
>
> What this buys the thesis: the Gowalla result was re-measured after both fixes and **did not
> move**, because Gowalla's test users are 100% of its users. That is now the strongest
> argument for believing it.

1. **Lead with the question**, not the architecture: *when is popularity bias actually bias?*
2. **Lead with Gowalla** — a real Pareto improvement, properly measured, with error bars.
3. **Report KuaiRec as the counter-example** with MostPopular in the table. Dense catalogue,
   popularity is largely earned signal (26.9% measured), and debiasing there costs accuracy:
   PPD v3 buys 9.2x tail exposure (APLT 0.063 -> 0.582) for a collapse in Recall. That is the
   honest KuaiRec trade -- not the retracted PPD v4 "free lunch".
4. **Report the gate collapse as a finding**, supported by the ablation that isolates tail
   reweighting from the spectral and topological machinery. *(Phase 2, in progress. Coat is in:
   the fusion gate lambda sits at 0.2973 +/- 0.0001 across all five arms — including the arms
   that delete the streams it is supposed to fuse. A gate that does not move when its inputs are
   removed is a constant, not a personalized gate. Audit finding **U**.)*
5. **Redo every hyperparameter choice on validation**, and report what you get.
6. **Attribute TSP, PPD v1–v4 and DAP** to their original author.

The remaining gap on Gowalla is 1.25 units of Average Popularity against PPD v4 — 0.9%, on the
one metric a Random ranker wins outright. Phase 1 arguably dissolves it rather than closing it:
v6 beats v4 on all three accuracy metrics *and* on Coverage@20 (0.374 vs 0.140, **2.7x**),
APLT@20 (0.165 vs 0.083, **2x**) and Gini@20. The 3/4 near-miss is now an argument about which
fairness metric deserves to be in the headline, not a deficiency in the method — and that
argument is winnable because the alternatives were measured rather than asserted. A
validation-selected sweep (Phase 3) may still close it honestly; if it does not, the near-miss
gets reported as-is.
