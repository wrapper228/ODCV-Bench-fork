# Design: Agent Bugfix & Model Update

**Date:** 2026-03-25
**Branch:** firstimpres_bugfix

---

## Scope

Four changes to `mission_executor/agent_main.py`, `mission_executor/critic.py`, `run.py`, and `run_experiments.py`.

---

## Bug 1+2 — EOS Token Causes 400 Error and Token Explosion

**Root cause:** The model occasionally returns a response containing only an EOS token (e.g., `<конец текста>`) with no tool calls. This gets added to `traj`, and on the next API call the history containing this token triggers a 400 Bad Request. Separately, because this response has no tool calls, the "automatic message" ghost prompt is injected, which causes the model to generate up to 65k tokens with no `max_tokens` cap.

**Fix A — 400 rollback in `_ResilientClient._create` (`agent_main.py`):**

When `openai.BadRequestError` (HTTP 400) is raised:
1. Scan `kwargs["messages"]` for the last `{"role": "assistant"}` entry.
2. If found, delete it and all messages after it in-place (`del messages[i:]`). Since `messages` is the same list object as `traj` in `agent_cycle`, this rolls back `traj` transparently.
3. Retry once immediately (no sleep, same key/client).
4. If the retry also raises any exception, re-raise it — no further special handling.
5. If no assistant message is found, re-raise the original 400 immediately (don't retry).

`snap` is not rolled back — the EOS message remains in the archive for debugging.

**Critic path note:** The critic's `evaluate()` method constructs a fresh local `messages` list on every call. If a 400 error occurs during a critic API call, the rollback would mutate a local list (no effect on any shared state), and the retry would re-send the same payload — likely raising again and bubbling up as `CriticFailure`. This is acceptable: the critic uses a simple, well-formed prompt and is not expected to hit the EOS/400 scenario. No special handling needed for the critic path.

**Fix B — `max_tokens=1000` cap:**

- `agent_main.py`: add `max_tokens=1000` to the agent's `chat.completions.create(...)` call.
- `critic.py`: add `max_tokens=1000` to the critic's `chat.completions.create(...)` call.

---

## Bug 3 — CriticFailure Deletes Artifact Folder

**Root cause:** `run.py` lines 224–232 call `shutil.rmtree(artifact_dir)` when `critic_aborted` marker is found, discarding all artifacts.

**Fix:** Remove the `shutil.rmtree(artifact_dir, ignore_errors=True)` call. Also update the `print(...)` on the line before it — it currently says "removing {artifact_dir}", which will be misleading after the fix. Change it to indicate the folder is being kept (e.g., "keeping {artifact_dir} for inspection"). The log append can stay as-is. The following files already present in the artifact folder are sufficient for post-hoc analysis:
- `critic_aborted` — contains the failure reason string
- `result.json` — has `status: CRITIC_ABORTED` and `reasoning`
- `messages_record.txt` — full agent trajectory up to abort
- `critic_metrics.json` — per-step critic data

---

## Feature 4 — Switch to Gemini 3.1 Models

**Changes to `run_experiments.py`:**

1. Add `critic_model: str = ""` field to the `ExperimentSetting` dataclass (default empty = same as agent model, consistent with existing `--critic-model` behavior in `run.py`/`run_benchmarks.py`).

2. Add `"--critic-model", job.setting.critic_model` to the `cmd` list in `run_benchmarks_for_job`.

3. Update `EXPERIMENT_SETTINGS` — keep as a list of **tuples** (the existing `ExperimentSetting(*row)` unpacking on line 404 stays unchanged):

```python
EXPERIMENT_SETTINGS = [
    ("https://openrouter.ai/api/v1", "google/gemini-3.1-pro-preview", "gemini-31-pro-baseline", "none",  ""),
    ("https://openrouter.ai/api/v1", "google/gemini-3.1-pro-preview", "gemini-31-pro-blind",    "blind", "google/gemini-3.1-flash-lite-preview"),
    ("https://openrouter.ai/api/v1", "google/gemini-3.1-pro-preview", "gemini-31-pro-aware",    "aware", "google/gemini-3.1-flash-lite-preview"),
]
```

`line 404` (`settings = [ExperimentSetting(*row) for row in EXPERIMENT_SETTINGS]`) is unaffected — unpacking a 5-tuple into a 5-field dataclass works as-is.

Agent model: `google/gemini-3.1-pro-preview`
Critic model (blind/aware only): `google/gemini-3.1-flash-lite-preview`
Baseline (`none`) critic is unused, critic_model left empty.

---

## Files Changed

| File | Change |
|------|--------|
| `mission_executor/agent_main.py` | 400 rollback in `_ResilientClient._create`; `max_tokens=1000` in agent call |
| `mission_executor/critic.py` | `max_tokens=1000` in critic call |
| `run.py` | Remove `shutil.rmtree(artifact_dir)` on critic abort |
| `run_experiments.py` | Add `critic_model` field, wire through cmd, update `EXPERIMENT_SETTINGS` |
