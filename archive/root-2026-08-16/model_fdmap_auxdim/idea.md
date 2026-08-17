# FDMap Model; Euclidean Distance with Auxiliary Dimensions

model_Euc_AugDim.py -- design notes + implementation plan (NOT YET
IMPLEMENTED). A future session should read this whole file before writing
any code, then implement STEP 1 first, validate it, and only then move to
STEP 2. Nothing below is code -- it's the complete record of a design
discussion, written so a fresh session with no memory of that discussion
can pick this up and act on it.


## -1. USER INPUT

    Similar to EuclideanDistanceModel (ForceDirected) in ./models.py but with
    auxiliary dimensions.

    The idea is to embed in a higher-dimensional space (d+aux dimensions) and
    gradually reduce the influence of the aux dimensions over time using
    a diminishing coefficient. This will help the model to escape local minima.

    It will try to embed in d+aux dimensions, but only the first d dimensions 
    will be used for the final embedding.
    The aux dimensions will be multiplied by a coefficient which diminishes
    over time.

    The coefficient is a function of epochs, and can be set by the user. The 
    following are some ideas:
    - A linear decay from 1 to 0: 
        coeff = linear_decay(epoch) = 1 - epoch / max_epochs
    - A cosine decay from 1 to 0:
        coeff = cosine_decay(epoch) = 0.5 * (1 + cos(pi * epoch / max_epochs))
    - A product of linear and cosine decay:
        coeff = linear_decay(epoch)*cosine_decay(epoch)
    - A sigmoid decay from 1 to 0:
        coeff = sigmoid_decay(epoch) = 1 / (1 + exp((epoch - max_epochs/2) / (max_epochs/10)))
    - A exponential decay from 1 to 0:
        coeff = exponential_decay(epoch) = exp(-epoch / (max_epochs/10))
    The user can also provide a custom function for the coefficient


## 0. STATUS

Idea + design analysis only. Zero lines of implementation exist yet.
This is deliberately a from-scratch research idea (not requested by any
existing test or caller) -- the point is to empirically find out whether
it helps, not to assume it does.


## 1. THE IDEA

Similar to ``EuclideanDistanceModel`` (``ForceDirected``) in ``./models.py``
but with auxiliary dimensions.

Embed in a higher-dimensional space (``d + aux`` dimensions) and gradually
reduce the influence of the ``aux`` dimensions over time using a
diminishing coefficient. The intent: give points "extra room" to route
around each other early in training (harder to get stuck behind a
topological obstruction in ``d+aux`` dims than in just ``d``), then
collapse that extra room away so the FINAL embedding is genuinely
``d``-dimensional. Only the first ``d`` dimensions are ever returned as
the final embedding; the ``aux`` ones are scaffolding.

The coefficient is a function of epoch (and the total epoch budget), and
should be pluggable by the caller. Candidate schedules (only ONE of these
should actually ship at first -- see Step 1's "keep it minimal" note):
    - linear:      coeff(epoch) = 1 - epoch / max_epochs
    - cosine:      coeff(epoch) = 0.5 * (1 + cos(pi * epoch / max_epochs))
    - linear*cosine product of the two above
    - sigmoid:     coeff(epoch) = 1 / (1 + exp((epoch - max_epochs/2) / (max_epochs/10)))
    - exponential: coeff(epoch) = exp(-epoch / (max_epochs/10))
All should map epoch 0 -> coeff ~= 1 and epoch ~= max_epochs -> coeff ~= 0.

## 2. WHY THIS MIGHT HELP (and why it's not a sure thing)

This is a real, if under-used, family of tricks in the MDS / force-directed
graph-layout literature -- related in spirit to "deterministic annealing"
and to t-SNE's early-exaggeration trick, though those anneal a FORCE
STRENGTH, not literal extra DIMENSIONS. Lifting into extra dimensions and
annealing them away has real precedent for escaping local minima in
low-target-dimension layouts (d=2/3), which is exactly the regime this
project's demo (``fdmap_jax_sell_c_sigma.py``, digits dataset) uses.

It is NOT a slam-dunk, though -- treat it as a hypothesis to test, not a
known win:
    - It's one of several tools for the same problem (alternatives:
      annealing repulsion strength directly, better initialization e.g.
      spectral/Laplacian-eigenmap init instead of random, multi-level /
      graph-coarsening approaches). This idea is worth trying, but hasn't
      been validated on this codebase's graphs yet.
    - It adds real hyperparameters (aux count, schedule, schedule
      parameters) that need tuning per dataset -- more surface area than
      the "no aux" baseline.
    - See section 4 for load-bearing tuning interactions that could make
      it perform WORSE than the baseline if not handled carefully.

Validation plan (do this for EVERY step below, not just once at the end):
compare final embedding quality with vs. without aux dims, same seed,
using something like the existing structure-recovery check pattern in
``validation/test_pipeline_behavior.py``'s ``test_end_to_end_structure_recovery``
(mean intra-shell distance ordering) or a stress/quality metric, across a
handful of seeds. A schedule that "sounds reasonable" but doesn't actually
improve on the baseline is a valid (negative) research result -- don't
skip this step just because Step 1 "runs and produces finite output."

## 3. IMPLEMENTATION PLAN -- two variants, implement and test IN ORDER

There are two materially different ways to apply the diminishing
coefficient. They have different engine impact, different failure modes,
and answer the "where does the coefficient get inserted" question very
differently. Do Step 1 FIRST (it's simpler, touches nothing shared), get
it empirically validated, THEN do Step 2 as a follow-up comparison -- not
because Step 2 is assumed better, but because it isolates a specific
hypothesis (see Step 2's rationale) that's only worth testing once Step 1
gives a real baseline to compare against.


--------------------------------------------------------------------
**STEP 1 (DO THIS FIRST): directly shrink Z's aux columns after each update**
--------------------------------------------------------------------
Mechanism: run the exact same force law over the full ``(n, d+aux)`` ``Z``
every epoch (no kernel changes at all), then, once per epoch, multiply
the aux columns of ``Z`` by ``coeff(epoch)``. As ``coeff -> 0`` the aux
columns collapse toward each other (all points converge to a shared,
~zero aux-coordinate), so their contribution to every pairwise distance
fades out on a common schedule. This is the variant the original idea
text above literally describes ("aux dimensions will be multiplied by a
coefficient").

Why start here: the SELL-C-sigma kernel (``embedding/sell_c_sigma.py``'s
``_step``) and ``EuclideanDistanceModel``'s force law are already
dimension-agnostic -- ``diff``, ``x = norm(diff)``, the padding contract,
all operate generically over whatever ``Z``'s last axis is. So this variant
needs ZERO changes to the engine. It's a pure model-level wrapper, using
the exact seam the codebase already documents for this kind of thing
(``core/force_directed.py``'s ``updateZ``/``updateGradient`` override
pattern -- same slot the momentum example there uses).

Exact insertion points (a NEW model class, e.g. subclassing
``EuclideanDistanceModel`` from ``./models.py`` -- suggested name
``EuclideanAuxDimModel``, but naming is not load-bearing):

1. ``__init__`` -- new constructor knobs: ``aux`` (int, number of auxiliary
   dims), ``decay_fn`` (callable ``(epoch, max_epochs) -> coeff in [0,1]``,
   default to exactly ONE of the schedules in section 1 -- see the
   "keep it minimal" note below), plus an internal epoch counter starting
   at 0. ``self.n_dim`` (inherited from ``ForceDirected``) should be set to
   ``d + aux``, not ``d`` -- the base class's lazy ``Z`` init in ``embed()``
   uses ``self.n_dim`` directly (``core/force_directed.py``), so this is
   the ONE place that controls Z's actual width; nothing else needs to
   know the model is "augmented."

2. ``embed()`` -- thin override, needed ONLY because ``updateZ(self,
   lr=None)`` doesn't receive ``epochs``, so there is nothing to normalize
   the schedule against unless it's captured explicitly:
       def embed(self, G, epochs=1000, **kwargs):
           self._max_epochs = epochs
           return super().embed(G, epochs=epochs, **kwargs)

3. ``updateZ()`` -- THE insertion point for the coefficient itself, right
   after the normal position update, so it composes cleanly with the base
   class's ``Z = Z + lr*dZ`` (and with any future momentum override of
   ``updateGradient`` -- these are independent seams, one on the force
   side, one on the integration side):
       def updateZ(self, lr=None):
           super().updateZ(lr=lr)             # Z = Z + lr*dZ, all d+aux columns
           self._epoch_counter += 1
           coeff = self.decay_fn(self._epoch_counter, self._max_epochs)
           self.Z = self.Z.at[:, self.d:].multiply(coeff)
   IMPORTANT: use a SELF-TRACKED counter, not ``self.latest_epoch`` --
   the base loop sets ``self.latest_epoch = epoch`` AFTER calling
   ``updateZ`` (see ``core/force_directed.py``'s ``embed()`` loop body),
   so reading ``self.latest_epoch`` from inside ``updateZ`` would be off
   by one. Incrementing our own counter here is exact regardless of that
   timing detail, and regardless of ``batch_count`` (``updateZ`` runs
   exactly once per epoch either way, independent of how many batches ran).

4. ``get_embeddings()`` -- companion piece, not the coefficient itself,
   but the other half of "only the first d dims are the real output":
       def get_embeddings(self):
           import numpy as np
           return np.asarray(self.Z[:, :self.d])

Known risks specific to this variant (test for these explicitly, don't
just eyeball the final picture):
    - TUG-OF-WAR: forces keep pushing the aux columns around every epoch
      based on the UNDAMPED physics; then the shrink step yanks them back
      down afterward. Depending on ``lr`` vs. the decay rate, this could
      (a) work as intended, (b) have the shrink dominate so fast the aux
      dims become irrelevant in a handful of epochs (defeating the
      purpose), or (c) oscillate. Plot ``||Z[:, d:]||`` over epochs during
      validation to see which regime you're actually in.
    - K1..K4 SCALE INTERACTION: early in training, pairwise distance ``x``
      is computed over the FULL ``d+aux`` dims, inflated by the randomly
      initialized aux columns. The force law's ``k1..k4`` defaults were
      tuned assuming ``x`` lives on the ``d``-dim scale -- they may need
      re-tuning per ``aux`` count, especially at small target ``d`` (e.g.
      ``d=2``, this project's visualization use case).
    - AUX INIT SCALE: today's lazy ``Z`` init (``jax.random.normal``,
      uniform across ALL columns, see ``core/force_directed.py``'s
      ``embed()``) gives the aux columns the same initial magnitude as the
      main columns, feeding directly into the risk above. Consider a
      SEPARATE (smaller) init scale for the aux columns as a first thing
      to try if the default init misbehaves -- would need its own small
      override of the lazy-Z-init path, since the base class doesn't
      expose a per-column init scale today.
    - EPSILON/TH INTERACTION: ``ForceDirected.Th(dZ)`` (the convergence
      statistic checked against ``epsilon``) is computed over the WHOLE
      ``dZ`` including aux columns. If the aux columns keep getting pushed
      by undamped forces and then externally reset every epoch, their
      contribution to ``dZ``'s norm may never settle -- this could prevent
      early convergence from ever triggering, or trigger on the wrong
      signal. If using ``epsilon``-based early stopping with this model,
      override ``Th`` to slice to ``[:, :self.d]`` first, and test both
      ways.
    - COMPUTE COST: every per-cell tensor in the bucketed engine scales
      with the full ``d+aux`` width. At ``d=128`` an ``aux=8`` is a ~6%
      overhead -- cheap. At ``d=2`` (this project's own digits-demo use
      case), ``aux=8`` is a 5x cost increase. Report this cost explicitly
      in whatever validation write-up comes out of testing this step.

NOTE: ship all five decay schedules. Set linear as default, 
plus a ``decay_fn=`` override for anyone who wants a different one. 
These functions are pretty simple, and the researcher can pick one or develop
one of their own.

--------------------------------------------------------------------
**STEP 2 (ONLY AFTER STEP 1 IS VALIDATED BY USER):**
**weight the aux dims' contribution to the distance metric, instead of shrinking Z itself**
--------------------------------------------------------------------
Rationale for even trying this: Step 1's tug-of-war risk (forces pushing
aux columns out, the shrink yanking them back in) is a real design smell.
This variant removes that tension by leaving ``Z``'s aux columns to evolve
freely under the UNDAMPED force law, and instead fading their INFLUENCE on
the pairwise distance metric used inside the force law:

    r = sqrt( sum_main(diff^2) + aux_scale(epoch)^2 * sum_aux(diff^2) )

As ``aux_scale -> 0``, the aux dims stop affecting ``r`` (and therefore
stop affecting the force magnitude) even though their raw values in ``Z``
never get reset -- mathematically cleaner, no shrink-vs-force tension,
but it does NOT produce a ``Z`` whose aux columns literally go to zero
(they're just inert by the end). Only try this once you have a working
Step-1 baseline to compare it against -- the entire point of doing this
second is to find out whether removing the tug-of-war actually helps
relative to Step 1, not to assume it's strictly better.

This variant is NOT a model-level wrapper -- it requires a real engine
change, because the coefficient has to reach inside the kernel:

    - A new TRACED argument into ``_step`` (``embedding/sell_c_sigma.py``),
      e.g. ``aux_scale`` -- passed at call time (like ``k1..k4``, like the
      recently-added ``h_shift``), NOT baked into the ``functools.partial``,
      so it can change every epoch without forcing a recompile.
    - A new STATIC split point (``d``, how many leading columns are
      "main") baked into the ``jax.jit`` via ``functools.partial`` --
      unlike ``aux_scale``, this determines a SLICE, which must be a
      trace-time constant, not a runtime value.
    - The ``diff = Zj - Zc[:, None, :]`` line inside ``_step`` would need
      to split into main-dims and aux-dims pieces (full weight vs.
      ``aux_scale``-weighted) before computing ``x = sqrt(sum(diff*diff))``.

This touches the ONE kernel every model in this package shares
(``ReferenceFDModel``, ``SparseFDModel``, ``EuclideanDistanceModel`` all
call into ``_step``) -- so this needs care to avoid regressing the other
models; the added arguments should default to "no-op" (``aux_scale=1.0``,
split point ``d = n_dim``) for every caller that isn't doing the
augmented-dimension experiment, mirroring how ``h_shift``/``h_data`` were
added as backward-compatible optional generalizations rather than
forcing every call site to change.

## 4. OPEN DESIGN QUESTIONS -- resolve these WHILE implementing, don't guess
   upfront and lock them in before the first experiment

- Decay schedule count: start with ONE (see Step 1's "keep it minimal"
  note), only add more if the first one demonstrably underperforms and a
  specific alternative is hypothesized to fix a specific failure mode.
- Should ``Th`` (the epsilon-convergence statistic) be computed over all
  ``d+aux`` dims or just the first ``d``, for this model specifically.
- Aux columns' init scale: same as main columns (simplest, current
  behavior) vs. a separate, smaller scale (needs a small override of the
  lazy-Z-init path in ``embed()``).
- Whether this ends up generic enough to be a reusable mixin over ANY
  ``ForceDirected`` subclass (not just ``EuclideanDistanceModel``) --
  the mechanism itself (Step 1 especially) doesn't actually depend on the
  Euclidean-distance force law at all. Worth revisiting once Step 1 is
  proven out: refactor into a mixin then, don't design one prematurely.
- Validation methodology: pick ONE concrete quality metric (e.g. the
  existing structure-recovery check's shape, or a stress measure) and
  compare with/without aux dims, same seeds, before drawing any
  conclusion from either step.

## 5. WHERE THINGS LIVE (for a session with no memory of this discussion)

- ``fdge_jax_sell_c_sigma/models.py`` -- ``ReferenceFDModel``,
  ``SparseFDModel``, ``EuclideanDistanceModel`` (composition root; the new
  augmented-dim model belongs here too, or in this file once it has real
  code).
- ``fdge_jax_sell_c_sigma/embedding/sell_c_sigma.py`` -- the bucketed
  engine: ``make_plan``, ``_step`` (the jitted per-epoch kernel),
  ``SellCSigmaForce``. Step 2 touches ``_step``; Step 1 touches nothing
  here.
- ``fdge_jax_sell_c_sigma/core/force_directed.py`` -- ``ForceDirected``
  base class: the ``embed()`` epoch loop, and the ``updateZ``/
  ``updateGradient`` override seams Step 1 uses.
- ``fdge_jax_sell_c_sigma/docs/DESIGN.md`` -- this package's design
  contract and import-discipline rules; keep any new model consistent
  with those (which stage packages it may import, etc.).
- ``fdge_jax_sell_c_sigma/validation/test_pipeline_behavior.py`` -- has
  the existing structure-recovery test pattern (``
  test_end_to_end_structure_recovery``) worth reusing/adapting for the
  validation plan in section 2.
