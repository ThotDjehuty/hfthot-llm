Symbolic Variational Calculus
==============================

Every stage of the pipeline — from the embedding projection in
:doc:`embeddings` to the LoRA update rule in :doc:`training` — is an instance
of the same underlying problem: minimising a functional over a space of
parameters or paths. This page makes that structure explicit and shows the
exact points where `optimiz-rs <https://optimiz-r.readthedocs.io/en/latest/>`_
supplies the CPU-only numerical primitives that make it fast.

Functionals and the first variation
------------------------------------

A **functional** :math:`\mathcal{J}[\theta]` maps a parameter vector (or a
whole trajectory) to a scalar. Training a neural network is functional
minimisation:

.. math::

   \theta^\star = \arg\min_{\theta \in \mathbb{R}^p} \mathcal{J}[\theta],
   \qquad
   \mathcal{J}[\theta] = \mathbb{E}_{(x,y)\sim\mathcal{D}}\bigl[\ell(f_\theta(x), y)\bigr]

where :math:`\ell` is the per-example loss (cross-entropy in
:doc:`training`) and :math:`\mathcal{D}` is the training distribution. The
**first variation** (Gâteaux derivative) of :math:`\mathcal{J}` in the
direction :math:`\delta\theta` is

.. math::

   \delta\mathcal{J}[\theta;\delta\theta]
   = \lim_{\varepsilon \to 0}
     \frac{\mathcal{J}[\theta + \varepsilon\,\delta\theta] - \mathcal{J}[\theta]}{\varepsilon}
   = \langle \nabla_\theta \mathcal{J}[\theta],\; \delta\theta \rangle.

A stationary point satisfies :math:`\delta\mathcal{J}[\theta;\delta\theta] = 0`
for all admissible :math:`\delta\theta` — the discrete analogue of the
**Euler–Lagrange equation**. For a Lagrangian :math:`L(t, \theta, \dot\theta)`
along a continuous parameter path :math:`\theta(t)`, stationarity of
:math:`\int_0^T L(t,\theta,\dot\theta)\,dt` requires

.. math::

   \frac{\partial L}{\partial \theta} - \frac{d}{dt}\frac{\partial L}{\partial \dot\theta} = 0.

Gradient descent as a gradient flow
------------------------------------

Discrete AdamW/SGD training is a forward-Euler discretisation of the
**gradient flow ODE**

.. math::

   \dot\theta(t) = -\nabla_\theta \mathcal{J}[\theta(t)],
   \qquad \theta(0) = \theta_0.

With step size :math:`\eta` and :math:`t_k = k\eta`, the explicit Euler
update :math:`\theta_{k+1} = \theta_k - \eta\,\nabla_\theta\mathcal{J}[\theta_k]`
recovers vanilla SGD; the LoRA update in :doc:`training`
(:math:`W' = W_0 + BA`) restricts this flow to the low-rank manifold
:math:`\{W_0 + BA : A \in \mathbb{R}^{r\times d},\, B \in \mathbb{R}^{d\times r}\}`,
so the Euler–Lagrange stationarity condition is enforced only along the
tangent space of that manifold — the same restricted-flow idea used by
Riccati-type reductions in ``optimiz_rs::control``.

This is exactly the viewpoint `optimiz-rs's BSDE module
<https://optimiz-r.readthedocs.io/en/latest/algorithms/bsde.html>`_ takes of
stochastic control: a forward SDE (here, noisy minibatch gradients) paired
with a backward value process. The Crank–Nicolson :math:`\theta`-scheme that
solves the linear BSDE

.. math::

   -dY_t = (a\,Y_t + b\,Z_t + c)\,dt - Z_t\,dW_t,
   \qquad Y_T = \xi,

second-order accurate in :math:`\Delta t`, is the same discretisation family
as the cosine learning-rate schedule (:doc:`training`) reinterpreted as a
terminal-value problem on the loss landscape: the schedule shapes
:math:`\eta_t` so the discrete flow's truncation error stays
:math:`\mathcal{O}(\Delta t^2)` near convergence, instead of the
:math:`\mathcal{O}(\Delta t)` of naive constant-step SGD.

Lagrangian view of mean-field self-training
---------------------------------------------

The ``llm-auto`` self-training loop (:doc:`orchestration`) repeatedly
retrains on a distribution of its own generated + session-corpus examples —
a discrete analogue of a **McKean–Vlasov** mean-field system, where the
"population" is the empirical distribution of training examples and the
drift depends on the *current* model's own output distribution:

.. math::

   d\theta_t = -\nabla_\theta \,\mathbb{E}_{x \sim \mu_t}\bigl[\ell(f_{\theta_t}(x))\bigr]\,dt,
   \qquad \mu_t = \text{Law}(x_t),\quad
   dx_t = b(x_t, \mu_t, f_{\theta_t})\,dt + \sigma\,dW_t.

`optimiz-rs's mean-field module
<https://optimiz-r.readthedocs.io/en/latest/algorithms/mean_field_games.html>`_
solves exactly this coupled forward (population) / backward (value)
fixed-point on CPU, via a Fokker–Planck forward sweep and a HJB backward
sweep — the numerical machinery generalises directly to the self-training
loop's fixed point between "current model" and "current training
distribution."

Evidence Lower Bound (ELBO) and functional derivatives
---------------------------------------------------------

The retrieval and privacy-audit tooling in :doc:`rag` and
:doc:`membership_inference` both rest on a variational approximation
:math:`q_\phi(z\mid x)` to an intractable posterior :math:`p(z\mid x)`. The
**evidence lower bound** is itself a functional of :math:`q_\phi`:

.. math::

   \mathcal{L}_{\mathrm{ELBO}}[\phi;x]
   = \mathbb{E}_{q_\phi(z\mid x)}\bigl[\log p_\theta(x\mid z)\bigr]
     - D_{\mathrm{KL}}\bigl(q_\phi(z\mid x)\,\|\,p(z)\bigr)
   \;\le\; \log p(x).

Taking the functional derivative of :math:`\mathcal{L}_{\mathrm{ELBO}}`
with respect to the density :math:`q_\phi` and setting it to zero (calculus
of variations over the space of densities, subject to
:math:`\int q_\phi = 1`) recovers the classical result that the *unconstrained*
optimum is :math:`q^\star(z\mid x) = p(z\mid x)` — the gap
:math:`\log p(x) - \mathcal{L}_{\mathrm{ELBO}}[\phi;x] = D_{\mathrm{KL}}(q_\phi\,\|\,p_\theta(z\mid x)) \ge 0`
measures exactly how far the parametric family :math:`q_\phi` falls short.
The reparameterisation trick

.. math::

   z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)

turns this into a pathwise (rather than score-function) gradient estimator,
so :math:`\nabla_\phi \mathcal{L}_{\mathrm{ELBO}}` can be computed by
ordinary backpropagation through the sampled :math:`z`. This is the
mechanism used by the VAE-based attack in :doc:`membership_inference`.

Why optimiz-rs makes this fast on CPU
----------------------------------------

Every functional-minimisation problem above reduces, numerically, to one of
three CPU-friendly primitives that optimiz-rs already implements natively
(no GPU, no Python interpreter overhead in the hot loop):

.. list-table::
   :header-rows: 1

   * - Functional problem
     - thotbook-AmentI use
     - optimiz-rs primitive
   * - Gradient flow / Euler–Lagrange descent
     - LoRA training loop (:doc:`training`)
     - ``optimiz_rs::control`` (Riccati / LQR reductions)
   * - Backward value process (BSDE)
     - Learning-rate scheduling as a terminal-value problem
     - ``linear_bsde_constant_coeffs`` — Crank–Nicolson :math:`\theta`-scheme,
       :math:`\mathcal{O}(\Delta t^2)`
   * - Mean-field fixed point
     - ``llm-auto`` self-training loop (:doc:`orchestration`)
     - ``solve_mfg_1d_rust`` — coupled Fokker–Planck / HJB sweep
   * - Global threshold calibration
     - Attack-threshold search (:doc:`membership_inference`)
     - ``differential_evolution`` — adaptive jDE, Rayon-parallel
   * - Two-sample distinguishability
     - Member vs. non-member score separation
     - ``mmd_gaussian``, ``mutual_information``

Because all of these are compiled Rust with no GPU dependency, the same
laptop that fine-tunes a LoRA adapter in :doc:`training` can also run the
attack-calibration and distinguishability tests in
:doc:`membership_inference` in seconds rather than minutes — the entire
functional-optimisation stack, from embedding to training loop to privacy
audit, stays CPU-only end to end.

See also
--------

- :doc:`training` — the discrete LoRA/AdamW update this page derives as a
  restricted gradient flow.
- :doc:`membership_inference` — the ELBO-based VAE attack built on the
  functional-derivative result above.
- `optimiz-rs: BSDE <https://optimiz-r.readthedocs.io/en/latest/algorithms/bsde.html>`_
- `optimiz-rs: Mean Field Games <https://optimiz-r.readthedocs.io/en/latest/algorithms/mean_field_games.html>`_
- `optimiz-rs: Stochastic Control <https://optimiz-r.readthedocs.io/en/latest/algorithms/stochastic_control.html>`_
