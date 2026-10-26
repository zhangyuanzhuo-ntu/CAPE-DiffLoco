# Reproducibility status and evidence checklist

**This repository is not yet a complete numerical reproduction of CAPE-DiffLoco.**

## Ready to inspect / test

- [x] CAPE abbreviation consistent with title: Contact-Aware Predictive Embedding.
- [x] GRU predictive encoder, `H=10`, `K=5`, latent dimension `16` in reference configuration.
- [x] Actor consumes encoder latent and current observation.
- [x] Fixed-alpha state alignment value/gradient unit tests.
- [x] Synthetic CPU smoke test exercises actor, encoder, alignment and surrogate gradient path.
- [x] Export script combines encoder and actor; TorchScript path can be smoke-tested.
- [x] Results scripts reject empty / fabricated evaluation templates.

## Must be supplied to reproduce paper evidence

- [ ] Exact Isaac Lab (and Isaac Sim) version, Go1 robot asset and terrain generator code, installation commands.
- [ ] Production environment adapter returning non-differentiable state, reset indices, feet contacts, true simulator-only supervision.
- [ ] Mathematically justified differentiable PD/Jacobian force mapping, impact/slip/contact surrogate, and multi-step rollout/BPTT.
- [ ] Observation schema with frame, signs, clipping, history timing, normalizer, and joint order.
- [ ] Ground-truth contact, support and slip supervision with timing/label definitions.
- [ ] End-to-end training and evaluation entry points using the actual Isaac environment.
- [ ] Independent PPO, original DiffSim and CAPE ablation training algorithms, all with matched compute budget.
- [ ] Terrain-specific held-out test scenes, random seeds, checkpoints and exact evaluation termination/success criteria.
- [ ] Episode-level raw simulation data and plots reproducibly regenerated from that data.
- [ ] Real Go1 EDU observation adapter, control mode switching, emergency stop, low-level control timing and hardware safety validation.
- [ ] Original real robot 20-trial-per-scenario records and videos, without synthetic or manually adjusted successes.

## What a smoke test can and cannot prove

The mock has a differentiable **action-to-force proxy** solely to show that gradient signal reaches policy parameters through state alignment. It has random synthetic contact labels, no calibrated PD/Jacobian, no valid physical contact mode or sim-to-real properties. Passing tests cannot validate the paper's quantitative claims or real-world performance.

Do not describe this repository as a complete implementation, production Isaac Lab trainer, verified Go1 deployment, or release of pretrained weights until those claims are supported by executable code and real logs.
