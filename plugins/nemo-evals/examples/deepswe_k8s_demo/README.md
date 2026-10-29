<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSWE on Kubernetes: run, watch, understand

[`deepswe_k8s_demo.ipynb`](deepswe_k8s_demo.ipynb) runs 10 DeepSWE tasks against a deployed NeMo Helix as a Kubernetes taskset run, shows each task's progress while you wait, then charts which tasks the agent solved, where the time went, and why a task failed.

![Dashboard: task timeline, time per phase, solved by language, tokens vs time](dashboard.png)

## What you need

- A notebook kernel whose environment has the `nemo` CLI with the evals plugin: `pip install 'nemo-helix[nemo-evals-plugin]'`, or this repository's `.venv`. The first code cell installs pandas and matplotlib into it if they're missing.
- A CLI context for the deployment: `nemo config set --context <name> --base-url <url>`. Set `CONTEXT` in the notebook's first code cell.
- A model the deployment's Inference Gateway serves (`MODEL`). The notebook puts a closed-book virtual model in front of it: DeepSWE tasks have no network, so the platform accepts a model-calling agent only through one.
- A sandbox profile, set by the platform admin in the Helm values (`evals.runs.harbor.sandbox_profiles`), whose `agent_setup_egress` lists the hosts codex installs from (Debian, Node.js, npm, GitHub). The admin can change a profile or add one; pick it with `SANDBOX_PROFILE`.
- Platform auth on: a secret holding a Scoped Access Key with `--scope inference`, named in `INFERENCE_KEY_SECRET`. Tested only with auth off so far.

## What it does

| Section | Commands |
| --- | --- |
| 1. Connect and prepare | `pip install pandas matplotlib`, `experiments create`, `inference virtual-models create` (each only if missing) |
| 2. Meet the tasks | Harbor Hub `task.toml` metadata: title, language, category |
| 3. Submit, `MODE = "taskset"` | `evals tasksets create` → `evals tasksets build-images --wait` → `evals runs submit --taskset` |
| 3. Submit, `MODE = "submit"` | `evals runs submit --harbor-hub` (publishes, builds and submits) |
| 4. Watch it run | `evals runs status --json`, every 15 s, as one tile per task |
| 5. Results | `evals runs download`, `jobs results download agent-eval-results` per task |
| 6. What happened | Headline numbers, 4 charts, generated findings, one failed task's drill-down |
| 7. Next | Studio leaderboard links; compare runs (`COMPARE`) |

- Every command is printed before it runs.
- Re-running cells in the same kernel returns the run already started instead of starting another; a new kernel starts a new run.
- `RUN_ID` skips submitting and analyzes an existing run.

## Time

All 10 tasks run at once, so the slowest task sets the pace.

- `codex`: up to about 40 min. Most tasks took 6–14 min of agent time; `fastapi-implicit-head-options` (16–34 min) and `expr-try-catch-errors` (19–21 min) were the slowest in our runs.
- `oracle` (each task's reference solution, no model): about 2 min. Use it to check the deployment, or as a ceiling next to an agent's run.
- Codex installs itself in each sandbox during setup. A trial that ends in an error, such as a failed install, reruns once in a new sandbox (`max_retries: 1`); a second failure ends it before the agent runs. The notebook reports those separately from tasks the agent didn't solve.

## Run it headless

```bash
papermill deepswe_k8s_demo.ipynb out.ipynb -p CONTEXT <context> -p AGENT oracle -p MODE submit
```
