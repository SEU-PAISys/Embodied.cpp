# Evaluation Configs

YAML files in this directory provide benchmark defaults for eval clients.
Command-line arguments always override values loaded from `--conf`.

Example:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf libero_pi05_eval.yaml
```

Run only one task from the same config:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf libero_pi05_eval.yaml \
  --task-id 0
```

Use explicit paths for machine-local assets such as tokenizers when needed.
