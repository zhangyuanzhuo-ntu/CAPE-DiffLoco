# CAPE-DiffLoco method definitions

**CAPE = Contact-Aware Predictive Embedding.**

Proprioceptive history and prediction:

\[
O_t=[o_{t-H+1},...,o_t],\qquad H=10,\quad (z_t,\hat C_{t:t+K})=f_\psi(O_t),\quad K=5,\quad z_t\in\mathbb R^{16}.
\]

Blind actor (12 joint targets):

\[a_t=\pi_\theta(o_t,z_t),\qquad a_t\in[-1,1]^{12}.\]

Contact-aware surrogate (idealized reference):

\[\dot v = m^{-1}\sum_{i=1}^4 s_i f_i + g,\qquad
\dot\omega=I^{-1}\left(\sum_i r_i\times s_if_i-\omega\times I\omega\right).\]

State alignment (forward high-fidelity value, backward surrogate derivative):

\[\bar x_{t+1}=x^{sim}_{t+1}+\alpha\left(x^{diff}_{t+1}-\mathrm{sg}(x^{diff}_{t+1})\right).\]

CAPE contact supervision may combine future contact BCE, support confidence regression, and slip-risk targets; these *require correctly timestamped, simulator-derived labels*. The mock implementation has only random synthetic contacts and cannot validate these losses. The correct Isaac Lab backward model remains an implementation milestone.
