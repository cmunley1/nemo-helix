<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSWE on Kubernetes: run, watch, understand

[`deepswe_k8s_demo.ipynb`](deepswe_k8s_demo.ipynb) runs 10 DeepSWE tasks against a deployed NeMo Helix as a Kubernetes taskset run, shows each task's progress while you wait, then charts which tasks the agent solved, where the time went, and why a task failed.

![Dashboard: task timeline, time per phase, solved by language, tokens vs time](dashboard.png)

## What you need

- The `nemo` CLI in the notebook kernel's environment: `pip install nemo-helix` (or this repository's `.venv`).
- A CLI context for the deployment: `nemo config set --context <name> --base-url <url>`. Set `CONTEXT` in the notebook's first code cell.
- A model the deployment's Inference Gateway serves (`MODEL`). The notebook puts a closed-book virtual model in front of it: DeepSWE tasks have no network, so the platform accepts a model-calling agent only through one.
- A sandbox profile, set by the platform admin in the Helm values (`evals.runs.harbor.sandbox_profiles`), whose `agent_setup_egress` lists the hosts codex installs from (Debian, Node.js, npm, GitHub). The admin can change a profile or add one; pick it with `SANDBOX_PROFILE`.
- Platform auth on: a secret holding a Scoped Access Key with `--scope inference`, named in `INFERENCE_KEY_SECRET`. Tested only with auth off so far.

## What it does

| Section | Commands |
| --- | --- |
| 1. Connect and prepare | `experiments create`, `inference virtual-models create` (only if missing) |
| 2. Meet the tasks | Harbor Hub `task.toml` metadata: title, language, category |
| 3. Submit, `MODE = "taskset"` | `evals tasksets create` → `evals tasksets build-images --wait` → `evals runs submit --taskset` |
| 3. Submit, `MODE = "submit"` | `evals runs submit --harbor-hub` (publishes, builds and submits) |
| 4. Watch it run | `evals runs status --json`, every 15 s, as one tile per task |
| 5. Results | `evals runs download`, `jobs results download agent-eval-results` per task |
| 6. What happened | Headline numbers, 4 charts, generated findings, one failed task's drill-down |
| 7. Next | Studio leaderboard links; compare runs (`COMPARE`) |

- Every command is printed before it runs.
- Re-running with the same parameters returns the same run (the idempotency key derives from them). Change `RUN_LABEL` for a new one.
- `RUN_ID` skips submitting and analyzes an existing run.

## Time

All 10 tasks run at once, so the slowest task sets the pace.

- `codex`: about 1 hour. Most tasks took 7–17 min of agent time; `dynamodb-toolbox-lazy-recursive-schemas` took 54–56 min in our runs. Drop it from `TASKS` for a faster demo.
- `oracle` (each task's reference solution, no model): about 2 min. Use it to check the deployment, or as a ceiling next to an agent's run.
- Codex installs itself in each sandbox during setup; a failed install (network) ends that trial before the agent runs. The notebook reports those separately from tasks the agent didn't solve.

## Run it headless

```bash
papermill deepswe_k8s_demo.ipynb out.ipynb -p CONTEXT <context> -p AGENT oracle -p MODE submit
```
