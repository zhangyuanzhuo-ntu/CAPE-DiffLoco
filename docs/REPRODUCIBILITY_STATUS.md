# Reproducibility status

| Component | Status |
|---|---|
| Terrain configuration and reproducible manifest generator | Implemented (parameters only; not Isaac scene generation) |
| PD and damped Jacobian-to-force mapping | CPU reference implemented, requires robot-specific Jacobians |
| Differentiable SRB and alignment | CPU reference implemented, simplified frames |
| Multi-step BPTT demonstration | Differentiable helper implemented; not validated on Isaac |
| History-only, reactive-contact, DiffSim, CAPE baseline architectures | Modules/design documented; full trainers not implemented |
| PPO full trainer | Not implemented; use a pinned supported PPO library after integration |
| Isaac Lab integration | Not implemented |
| Real-world evaluation | Not released; publish only actual measured logs |
| Go1 joystick command adapter | Architecture documented; hardware integration not implemented |
| Pretrained weights | Not released |

No hardware experiments or training metrics are asserted by this extension.
