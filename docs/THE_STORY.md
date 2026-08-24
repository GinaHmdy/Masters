# The Whole Story
### Making graph recommenders fairer without making them worse — what we tried, what worked, and what didn't

*Written for a computer science student who has never worked on recommender systems.
No prior knowledge assumed. Read it top to bottom.*

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

---

## Part 3 — What already existed when this work started

This matters, and thesis write-ups often blur it. Three components were **already in the
codebase**, written by a previous author, before this work began:

| Component | What it does | Author |
|---|---|---|
| **TSP** (Topological Simplex Propagation) | smooths embeddings over a higher-order structure | inherited |
| **PPD v1–v4** (Popularity/Preference Debiasing) | removes a "popularity direction" from embeddings | inherited |
| **DAP** (Degree-Aware Propagation) | clusters nodes and subtracts high/low-degree anchors | inherited |

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

### 5.3 KuaiRec (dense): the most instructive failure

The write-up's headline is a **32.8× improvement** in Recall over LightGCN.

Whenever you see a number like that, ask one question: *what is the baseline doing?*

The LightGCN baseline on KuaiRec scores Recall@20 = 0.00074. Its Recall@5 is **exactly
zero** — in five attempts across 1,410 users it never once got a single item right. That's
not a baseline. That's a broken training run. Dividing by it produces an impressive ratio and
tells you nothing.

So here is the check that should have been run on day one. Ignore personalization entirely.
Rank items by raw popularity. Recommend the same 20 items to everybody:

| Dataset | MostPopular Recall@20 | Best model in the thesis |
|---|---|---|
| Gowalla | 0.040822 | 0.142838 ✓ 3.5× better |
| Yahoo | 0.090639 | 0.142360 ✓ 1.6× better |
| Coat | 0.095386 | 0.125553 ✓ 1.3× better |
| **KuaiRec** | **0.123342** | **0.013903 ✗ 9× worse** |

**On KuaiRec, recommending the same 20 popular items to every single user beats every model
in this thesis by roughly nine times.**

That sounds humiliating. It isn't — it's the most scientifically interesting result in the
project, provided you report it. Look at why it happens: KuaiRec's test set has about **103
relevant items per user** out of 10,728. It is extraordinarily dense. When a user likes 1% of
the entire catalogue, popular items really are the best guess, and *popularity is not a bias
here — it is the signal.*

Which reframes the whole thesis question. Not *"how do we remove popularity bias?"* but:

> **"When is popularity bias actually bias, and when is it information?"**

That's a better question, and this project has the data to answer it.

> **Lesson 5.** Always run the dumbest possible baseline first. If it wins, you've learned
> something important about your data.

### 5.4 Coat (small): no

PPD v6 loses to every baseline on every accuracy metric by roughly 2 standard deviations.
Coat is quietly missing from the results section of the project write-up, while still being
counted in the phrase "four diverse datasets". If a dataset is in your claim, it's in your
table.

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
| λ | the personalized fusion gate | **0.4998** — exactly its starting value |

Read that carefully:

- θ ≈ 0 means the **topological stream was switched off** by training. Ingredient 1 of the
  architecture: inert.
- α ≈ 0 means **almost no popularity was being removed** by the spectral projection.
  Ingredient 2, the QR subspace — the mathematical centrepiece: inert.
- λ stuck at exactly 0.4998 means the "personalized" gate **never learned anything** — it sat
  at the 0.5 its sigmoid starts at. Ingredient 4: not personalized at all, just a constant.

Gradient descent was handed these mechanisms and quietly turned all three off.

**So what produced the popularity reduction on Gowalla?** The one ingredient that doesn't
depend on θ or α: **the tail reweighting** — `(1 − popularity)^φ` at scoring time, and extra
weight on tail items in the loss. Ingredient 3. The simplest thing in the whole design.

This is a genuinely valuable finding. Restated for a thesis:

> I built a dual-stream architecture with a spectral popularity subspace, a topological
> routing gate and a personalized fusion gate. I instrumented all three. On dense data all
> three collapse to zero, and the entire measurable effect comes from simple degree-based
> tail reweighting.

That is *more* interesting than "my complicated model won by 0.8%". Negative results about
your own mechanisms, backed by instrumentation, are real science. The thesis just has to
claim it deliberately rather than have an examiner discover it in the logs.

> **Lesson 6.** Instrument your mechanisms and check they're actually running. A model can
> improve for reasons that have nothing to do with your idea.

---

## Part 7 — The seven lessons

1. **Structure derived from your own model teaches it nothing.** (TSP)
2. **After deleting a term, hunt down its parameter.** (dead `--beta`)
3. **Choose on validation. Touch test once.** (Yahoo)
4. **A difference without an error bar is not a result.**
5. **Run the dumbest baseline first.** (KuaiRec)
6. **Instrument your mechanisms — verify they're switched on.** (θ, α, λ)
7. **Say what you inherited.** Nothing is lost by attributing prior work, and everything is
   lost by not.

---

## Part 8 — Where this should go next

The clean version of this thesis is smaller and much stronger:

1. **Lead with the question**, not the architecture: *when is popularity bias actually bias?*
2. **Lead with Gowalla** — a real Pareto improvement, properly measured, with error bars.
3. **Report KuaiRec as the counter-example** with MostPopular in the table. Dense catalogue,
   popularity is signal, debiasing correctly costs accuracy.
4. **Report the gate collapse as a finding**, supported by the ablation that isolates tail
   reweighting from the spectral and topological machinery.
5. **Redo every hyperparameter choice on validation**, and report what you get.
6. **Attribute TSP, PPD v1–v4 and DAP** to their original author.

The remaining gap on Gowalla is 1.25 units of Average Popularity. One honestly-selected run
could close it. That's a good place to be.
