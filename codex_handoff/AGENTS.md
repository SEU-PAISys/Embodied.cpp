# Repository Instructions

## User and purpose

The user is a second-year electronic engineering student learning Python,
PyTorch, C++, CUDA, AI infrastructure, and Embodied.cpp. Optimize for both
correctness and learning. Do not optimize only for finishing the task quickly.

Read these files when relevant:

- `codex_handoff/HANDOFF_SUMMARY.md`
- `codex_handoff/docs/04_TASK_BACKLOG.md`
- `codex_handoff/docs/05_PROMPT_LIBRARY.md`
- `codex_handoff/docs/06_ENVIRONMENT_AND_REPO_RULES.md`

## Before changing code

For every non-trivial task:

1. Restate the smallest verifiable goal.
2. Identify the files and functions likely involved.
3. Explain expected inputs, outputs, tensor shapes, and ownership where relevant.
4. Present a short verification plan.
5. Ask for confirmation before changing more than three source files.

For repository exploration, start read-only and report evidence with file paths
and symbol names. Do not edit while mapping architecture.

## Change discipline

- Make the smallest reversible change.
- Do not refactor unrelated code.
- Do not edit vendored or third-party code unless explicitly requested.
- Do not disable tests or suppress warnings to make a build pass.
- Do not install packages globally.
- Do not modify GPU drivers, Linux display drivers, system Python, or shell
  startup files without explicit approval.
- Do not delete files, run destructive Git commands, rewrite history, or force
  push.
- Never expose secrets or include model weights, checkpoints, build products,
  datasets, or credentials in commits.

## Debugging

1. Reproduce the failure.
2. Find the first real error, not later cascading errors.
3. State root cause, evidence, minimal fix, and verification.
4. Prefer `-j1` when capturing an unreadable C++ build failure.
5. Preserve the failing command and relevant environment details.

## C++ and CUDA

Explain:

- object lifetime and ownership;
- pointer/reference semantics;
- CPU/GPU synchronization;
- warm-up and timing boundaries;
- whether timing includes memory transfer;
- possible undefined behavior and thread-safety risks.

For performance claims, use repeated measurements and report mean, standard
deviation, p50, p95, and p99 when appropriate.

## Python and tensors

- Always state important tensor shapes and dtypes.
- Use a project-local virtual environment.
- Do not silently change package versions.
- Keep random seeds and experiment configuration explicit.
- Separate training, evaluation, and benchmarking.

## After changing code

1. Show a concise summary of each changed file.
2. Run the narrowest relevant build/test first.
3. Report exact commands and results.
4. State remaining risks and untested paths.
5. Ask the user to inspect `git diff` before committing.
6. Provide two or three questions the user should be able to answer to confirm
   understanding.
