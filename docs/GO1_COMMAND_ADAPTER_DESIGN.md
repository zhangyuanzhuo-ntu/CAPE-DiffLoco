# Go1 EDU command adapter — design only

Do **not** modify the radio controller firmware or Unitree SDK motor indices.
Read the wireless remote state through the compatible Go1 low-level SDK and
translate calibrated joystick axes into `(vx_cmd, vy_cmd, yaw_rate_cmd)`.
Validate the axis signs and scaling on the actual remote before use.

Pipeline:
1. Remote decoder and deadman/enable switch.
2. Command dead-zone, slew-rate limiter, bounds matching training distribution.
3. Observation adapter (IMU frames, 12 joint positions/velocities, joint order).
4. 10-step history buffer -> CAPE encoder -> actor -> 12 target offsets.
5. Joint mapping and sign/offset calibration -> command bounds -> low-level SDK.
6. Independent safety supervisor and physical emergency-stop mechanism.

Only one locomotion controller may own the motor command channel at a time.
Exit vendor sport mode using the documented procedure for the exact Go1 firmware
before low-level control. Do not assume a high-level `HighCmd` speed request
will run CAPE; that invokes the stock controller.

A model file (.pt or .onnx) is not executable control software. The Go1 onboard
computer needs a compatible inference runtime, command bridge, low-level loop,
watchdog, telemetry logger, and tested failure recovery. Model export should
bundle encoder + actor + observation normalization metadata.

Required hardware gates:
- Software-only joint permutation and unit tests
- Suspended robot, low gain, limited movement, operator present
- Latency and packet loss tests; remote disconnect and watchdog tests
- Safety-supervised flat-ground validation before terrain traversal

**Not implemented here:** Go1-specific UDP control, remote packet decoder,
control arbitration, safe shutdown, and hardware validation. These require
firmware-specific testing and must not be represented as finished.
