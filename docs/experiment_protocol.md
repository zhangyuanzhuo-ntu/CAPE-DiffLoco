# Experimental protocol — planned, not yet implemented

## Simulation

Training domains: gap crossing, mixed terrains, rough terrain, regular stairs. Hold **unseen** stair geometries out of training; do not call a training terrain "unseen" without documenting the split.

Report robot asset/version, actuator gains, time step, policy rate, observation/action definition, command ranges, terrain generator distributions, domain randomization, seeds, number of environments, optimizer, gradient horizon and training budget. Use identical scenario definitions and evaluation budgets for all baselines.

## Real-world tests

Planned Go1 EDU scenarios: indoor stairs; iron-plate stairs; bumpy grass/mud; tall grass/mud; 40-degree ascent; 40-degree descent; wet brick stairs; grooves. For each method and scenario, collect 20 or more **actual independent** trials with success, fall, intervention, traversal bounds, test date and environmental conditions. Per-scenario success rates for exactly 20 trials change in increments of 5 percentage points.

## Measurements

- Success/fall rate per method, scenario, trial; define termination and external intervention.
- Translational velocity tracking and angular velocity / heading tracking; label yaw-rate error vs yaw-angle error unambiguously.
- Contact F1 computed from thresholded probabilities and real labels (not random mock labels).
- Slip incidence, recovery latency, roll/pitch stability, energy and action safety bounds if measured.

Do not use mock smoke-test losses or invented raw rows as quantitative empirical evidence.
