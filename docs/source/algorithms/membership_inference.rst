Membership Inference — Privacy Audit
======================================

thotbook-AmentI is *"private by design"*: content is redacted at ingest
time and a write-time leak audit refuses to append rows that still contain
secrets (see the project README). **Membership inference** is the
complementary check on the *output* side — given a trained model, can an
observer tell whether a specific document was part of its training set?
This page documents the attack methodology thotbook-AmentI ships as a
**self-audit tool**: you run it against your own locally trained LoRA
adapter to quantify how much it memorises, before you ever share weights,
embeddings, or a hosted endpoint with anyone else.

.. warning::

   This is a defensive/audit tool, not an attack framework for third-party
   models. Run it only against models you trained yourself, on your own
   corpus, to measure your own privacy exposure — exactly the workflow in
   the companion notebook (:ref:`mia-notebook`).

Threat model
------------

Let :math:`f_\theta` be the trained model, :math:`\mathcal{D}_{\mathrm{tr}}`
its training set, and :math:`x` a candidate document. A **membership
inference attack** (MIA) is a function

.. math::

   \mathcal{A}: (f_\theta, x) \longmapsto s(x) \in \mathbb{R},
   \qquad
   \widehat{m}(x) = \mathbb{1}[s(x) > \tau]

that outputs a real-valued score :math:`s(x)` and thresholds it at
:math:`\tau` to predict membership :math:`\widehat m(x) \in \{0,1\}`. We
evaluate :math:`\mathcal{A}` on a balanced set of *known* members
(:math:`x \in \mathcal{D}_{\mathrm{tr}}`) and *known* non-members
(held-out :math:`x \notin \mathcal{D}_{\mathrm{tr}}`), and report the
**ROC curve** of :math:`\widehat m` as :math:`\tau` varies, its **AUC**,
and — the metric that actually matters for privacy, per Carlini et al.
(2022) — the **true-positive rate at low false-positive rate**
(:math:`\mathrm{TPR}@\mathrm{FPR}{=}0.01`), since an attacker who can
confidently flag even a handful of members at near-zero false-positive
rate has a real privacy break even if the overall AUC looks unremarkable.

Baseline 1 — Loss-threshold attack (Yeom et al., 2018)
--------------------------------------------------------

The simplest attack exploits the fact that a model's per-example loss is,
on average, lower on data it was trained on:

.. math::

   s_{\mathrm{loss}}(x) = -\ell\bigl(f_\theta(x), y\bigr)

i.e. score = negative cross-entropy loss (:doc:`training`); the model is
"more confident," hence "more likely a member," on low-loss examples.
This requires only black-box query access to the loss and is the floor
every stronger attack in this page is compared against.

Baseline 2 — Likelihood-ratio attack (LiRA)
----------------------------------------------

Carlini et al.'s **LiRA** trains :math:`k` *shadow models*
:math:`f_{\theta_1}, \dots, f_{\theta_k}` on resampled datasets that each
either include or exclude :math:`x`, fits Gaussians to the two resulting
loss populations

.. math::

   \ell(f_{\theta_i}(x)) \mid x \in \mathcal{D}_{\theta_i}
     \sim \mathcal{N}(\mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2),
   \qquad
   \ell(f_{\theta_i}(x)) \mid x \notin \mathcal{D}_{\theta_i}
     \sim \mathcal{N}(\mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2),

and scores the target model's loss :math:`\ell(f_\theta(x))` by the
log-likelihood ratio

.. math::

   s_{\mathrm{LiRA}}(x)
   = \log \frac{\mathcal{N}(\ell(f_\theta(x)); \mu_{\mathrm{in}}, \sigma_{\mathrm{in}}^2)}
               {\mathcal{N}(\ell(f_\theta(x)); \mu_{\mathrm{out}}, \sigma_{\mathrm{out}}^2)}.

This is a **calibrated**, per-example attack (it accounts for the fact
that some documents are intrinsically "easy," hence low-loss, for every
model regardless of membership) and is the strongest of the classical
baselines at low FPR.

Agent-agnostic VAE attack
----------------------------

Both baselines above need repeated access to :math:`f_\theta`'s loss —
white/grey-box access tied to *one specific* model architecture. The
**VAE attack** shipped here is deliberately **agent-agnostic**: it is
trained once on a *generic feature vector*

.. math::

   \phi(x) = \bigl[\, \hat{\mathbf{e}}(x),\; \ell(x),\; \mathrm{rank}(x),\; H(x) \,\bigr]
   \in \mathbb{R}^{d+3}

— the L2-normalised embedding :math:`\hat{\mathbf{e}}(x)` from
:doc:`embeddings`, the per-example loss, its rank among a reference batch,
and the token-level entropy :math:`H(x)` — none of which are specific to
Qwen3-8B, LoRA, or even a transformer: any model that can emit an
embedding and a scalar loss plugs into the same attack. A VAE
:math:`(q_\phi, p_\theta)` is fit **only on the member population's**
feature vectors, using the ELBO derived in
:doc:`variational_calculus`:

.. math::

   \mathcal{L}_{\mathrm{ELBO}}[\phi;x]
   = \mathbb{E}_{q_\phi(z\mid \phi(x))}\bigl[\log p_\theta(\phi(x)\mid z)\bigr]
     - D_{\mathrm{KL}}\bigl(q_\phi(z\mid \phi(x))\,\|\,\mathcal{N}(0,I)\bigr).

Members — the distribution the VAE was fit on — reconstruct with higher
ELBO (lower reconstruction error); non-members, being out-of-distribution
for the fitted :math:`q_\phi`, reconstruct worse. The attack score is the
ELBO itself:

.. math::

   s_{\mathrm{VAE}}(x) = \mathcal{L}_{\mathrm{ELBO}}[\phi; \phi(x)].

Because :math:`\phi(x)` never exposes raw model weights or gradients, this
scorer can be *shared or reused across agents/models* without leaking the
underlying network — hence "agent-agnostic": the same fitted VAE that
audits the Qwen3-8B LoRA adapter can, unmodified, audit a differently
architected model, as long as it can produce an embedding and a loss.

Threshold calibration and evaluation
----------------------------------------

The decision threshold :math:`\tau` that maximises Youden's J statistic
:math:`J(\tau) = \mathrm{TPR}(\tau) - \mathrm{FPR}(\tau)` is found by a
1-D global optimisation — ``optimiz_rs::differential_evolution`` — rather
than a linear scan, so the same calibration step that takes milliseconds
here scales unchanged to a multi-dimensional threshold surface if
:math:`s(x)` is later replaced by a learned combination of the four
scores above.

Distinguishability between the member-score and non-member-score
distributions is additionally tested with two model-free statistics from
``optimiz-rs``:

.. math::

   \mathrm{MMD}^2(P_{\mathrm{in}}, P_{\mathrm{out}})
   = \mathbb{E}[k(s,s')] - 2\,\mathbb{E}[k(s,\tilde s)] + \mathbb{E}[k(\tilde s,\tilde s')],
   \qquad k(a,b) = e^{-\|a-b\|^2/2\sigma^2}

(``mmd_gaussian``, :math:`s \sim P_{\mathrm{in}}`,
:math:`\tilde s \sim P_{\mathrm{out}}`), and the mutual information
:math:`I(s(x); m(x))` between the attack score and the true membership
label (``mutual_information``) — both zero iff the attack carries no
signal at all, giving a sanity floor independent of any particular
threshold choice.

Uncertainty is reported as a **block-bootstrap confidence interval**
(block length :math:`\ell \approx 21`, :math:`B \ge 2000` resamples, per
this workspace's companion-notebook statistics convention) on the AUC of
each attack, so that "attack A beats attack B" claims are only made when
the CIs are non-overlapping.

.. _mia-notebook:

Companion notebook
-------------------

The full pipeline — real ``thotbook train`` LoRA runs on member/non-member
splits, real per-example loss and embedding extraction, the four scores
above, ROC/AUC with block-bootstrap CIs, and the VAE agent-agnostic
scorer — is implemented end to end, with real (not mocked) calls into the
compiled Rust crates, in
``notebooks/membership_inference_audit.ipynb``.

See also
--------

- :doc:`variational_calculus` — derivation of the ELBO used by the VAE
  scorer.
- :doc:`training` — the loss function the loss-threshold and LiRA
  baselines read from.
- :doc:`embeddings` — the feature :math:`\hat{\mathbf{e}}(x)` used by the
  agent-agnostic scorer.
- Shokri, R. et al. (2017). *Membership Inference Attacks Against Machine
  Learning Models*. IEEE S&P.
- Yeom, S. et al. (2018). *Privacy Risk in Machine Learning: Analyzing the
  Connection to Overfitting*. IEEE CSF.
- Carlini, N. et al. (2022). *Membership Inference Attacks From First
  Principles*. IEEE S&P.
