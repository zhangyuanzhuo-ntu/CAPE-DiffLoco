# Production Isaac Lab integration contract (not implemented)

The mock runner is **not** an Isaac Lab wrapper and must not be used for benchmark results.

1. **Simulator backend.** Pin Isaac Sim and Isaac Lab versions; supply an installable Go1 articulation asset, terrain generators (gap, mixed, rough, unseen stairs) and explicit physics / policy substep frequencies.
2. **Observations.** Provide velocity commands, body-frame gyro, projected gravity, 12 joint positions and velocities, previous action, and estimated contact observations in the exact order, with normalization and delay behavior identical to deployment. The stated 49 dimensions include four estimated contact channels; verify with training artifacts.
3. **Training-only labels.** Return true foot contact states, support probabilities (with defensible labels), friction/slip targets and terrain-relative foot clearance from simulator. Predict future labels over `K=5` control steps, using correctly aligned timestamps and episode reset masking.
4. **Forward dynamics.** Step Isaac Lab using joint targets and PD gains; read detached ground-truth next state `x_sim` in a documented 13D state convention.
5. **Backward dynamics.** Implement a real differentiable SRB step with action-dependent, physically justified foot forces and contact confidence. The mock `forces = nominal + 5*action.reshape(...)` **must never be used as the scientific surrogate**.
6. **Alignment and losses.** Use the aligned state in task losses. Verify gradients from tracking/posture to both encoder and actor, compare against fixed-alpha original DiffSim, apply short-horizon gradient clipping and truncation deliberately.
7. **Baselines and evaluation.** Train PPO, original DiffSim, history-only, reactive contact and ablations independently. Log actual per-episode seeds, terrain, commands, success, fall, yaw/velocity errors, contacts and recovery metrics. Evaluate on held-out scenes.
8. **Hardware.** Go1 EDU low-level SDK/ROS2 runner, observation order checks, safe control arbitration and emergency-stop verification must precede real locomotion.

`python scripts/train.py --config configs/train_cape_go1.yaml` must currently fail immediately, **not** silently train the mock model. This is intentional reproducibility protection.
