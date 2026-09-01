# LIBERO Evaluation Reports

Unified close-out of the three VLA runtimes integrated into Embodied.cpp in
this cycle: **TurboVLA**, **Xiaomi-Robotics-0** and **X-VLA**.
Every runtime is validated against its official PyTorch implementation on
LIBERO before being declared done.

| Model | Params | Scope | Official ref | Ours (best) | Verdict |
|---|---|---|---|---|---|
| [TurboVLA](turbovla_libero.md) | 0.2B | full LIBERO, 400 eps | 97.7% avg | **96.5%** (q8_0) / 96.25% (bf16) | within ~1 pp of official |
| [Xiaomi-Robotics-0](xr0_libero.md) | 4.7B | full LIBERO, 2000 eps/config | 98.7% avg | **98.45%** (bf16) / 98.55% (q4_k) | matches official |
| [X-VLA](xvla_libero.md) | 0.9B | full LIBERO, 400 eps | 99.8% | **99.2%** (bf16) / 99.8% (q8_0) | matches official |

Upstream snapshot at time of writing:

| Upstream | Status |
|---|---|
| SEU-PAISys/Embodied.cpp `main@c5a96a2` | no new upstream commits since 2026-08-12 |
| XiaomiRobotics/Xiaomi-Robotics-0 | latest: post-training code open-sourced 2026-04-27 |
| H-EmbodVis/TurboVLA | latest: checkpoints released 2026-07-31 (LIBERO avg 97.7%) |
| 2toinf/X-VLA | accepted to ICLR 2026; natively integrated into LeRobot |

Raw episode logs and videos are kept out of git (`outputs/` is ignored);
the tables in these documents are generated from those logs.
