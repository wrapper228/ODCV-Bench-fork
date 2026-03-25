# Agent Bugfix & Model Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix EOS-token 400 errors and token explosion, preserve artifacts on CriticFailure, and switch to Gemini 3.1 models with a separate cheaper critic model.

**Architecture:** Four targeted changes across `agent_main.py`, `critic.py`, `run.py`, and `run_experiments.py`. No new files. The 400 rollback is the only behaviorally complex change; the rest are one-liners or small additions.

**Tech Stack:** Python 3.10+, openai SDK, pytest, Docker Compose

---

## File Map

| File | Change |
|------|--------|
| `mission_executor/agent_main.py` | 400 rollback in `_ResilientClient._create`; `max_tokens=1000` on agent call (line ~286) |
| `mission_executor/critic.py` | `max_tokens=1000` on critic call (line ~188) |
| `run.py` | Remove `shutil.rmtree(artifact_dir)` (line 232); update print/log strings |
| `run_experiments.py` | Add `critic_model` field to `ExperimentSetting`; wire `--critic-model` in cmd; update `EXPERIMENT_SETTINGS` |
| `tests/test_resilient_client.py` | Add 4 tests for 400 rollback behavior |

---

## Task 1: 400 Rollback in `_ResilientClient` (TDD)

**Files:**
- Modify: `tests/test_resilient_client.py`
- Modify: `mission_executor/agent_main.py:79-108`

### Step 1.1 — Write failing tests

Add the following to `tests/test_resilient_client.py` (after the existing tests). Note: `_make_response()` is already defined near the top of the file — these tests reuse it.

```python
# ---------------------------------------------------------------------------
# 400 Bad Request rollback tests
# ---------------------------------------------------------------------------

class _BadRequest(Exception):
    """Minimal stand-in for openai.BadRequestError (has status_code=400)."""
    status_code = 400


def test_400_rolls_back_last_assistant_and_retries():
    """400 error: last assistant + subsequent messages deleted in-place, single retry succeeds."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = [_BadRequest("bad"), _make_response()]

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<конец текста>"},
        {"role": "user", "content": "ghost"},
    ]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=messages)

    assert resp.choices is not None
    assert mock_inner.chat.completions.create.call_count == 2
    # messages list mutated: assistant + ghost removed
    assert len(messages) == 2
    assert messages[-1] == {"role": "user", "content": "task"}


def test_400_no_assistant_message_raises_without_retry():
    """400 with no assistant message in history: raises immediately, no retry."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = _BadRequest("bad")

    messages = [{"role": "user", "content": "task"}]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        try:
            client.chat.completions.create(model="m", messages=messages)
            assert False, "should have raised"
        except Exception as exc:
            assert "bad" in str(exc).lower()

    assert mock_inner.chat.completions.create.call_count == 1


def test_400_rollback_retry_fails_raises_retry_error():
    """400 rollback attempted, retry also fails: raises retry error, only 2 calls total."""
    mock_inner = MagicMock()
    mock_inner.chat.completions.create.side_effect = [
        _BadRequest("bad"),
        Exception("still broken"),
    ]

    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "<eos>"},
    ]

    with patch("agent_main.OpenAI", return_value=mock_inner), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        try:
            client.chat.completions.create(model="m", messages=messages)
            assert False, "should have raised"
        except Exception as exc:
            assert "still broken" in str(exc)

    assert mock_inner.chat.completions.create.call_count == 2


def test_400_does_not_fall_through_to_backup_key():
    """After 400 handling (even if it fails), backup key is never tried."""
    primary = MagicMock()
    backup = MagicMock()
    primary.chat.completions.create.side_effect = [_BadRequest("bad"), _make_response()]

    def make_client(base_url, api_key):
        return primary if api_key == "pk" else backup

    messages = [{"role": "user", "content": "t"}, {"role": "assistant", "content": "<eos>"}]

    with patch("agent_main.OpenAI", side_effect=make_client), \
         patch("agent_main.time"):
        client = _ResilientClient("http://base", "pk", "bk")
        resp = client.chat.completions.create(model="m", messages=messages)

    assert resp.choices is not None
    assert primary.chat.completions.create.call_count == 2
    assert backup.chat.completions.create.call_count == 0
```

- [ ] **Step 1.2 — Run tests to confirm they fail**

```bash
cd /c/Users/atama/Documents/GitHub/ODCV-Bench-fork
python -m pytest tests/test_resilient_client.py -k "400" -v
```

Expected: 4 FAILED (AttributeError or AssertionError — `_create` has no 400 handling yet)

- [ ] **Step 1.3 — Implement 400 rollback in `_ResilientClient._create`**

Replace the `_create` method body in `mission_executor/agent_main.py` (lines 79–108). The new implementation adds a `handled_400` flag and special-cases `status_code == 400` inside the `except` block:

```python
def _create(self, **kwargs):
    keys_and_waits = [(self._primary_key, self._PRIMARY_WAITS)]
    if self._backup_key:
        keys_and_waits.append((self._backup_key, self._BACKUP_WAITS))

    last_exc: Exception | None = None

    for key_idx, (key, waits) in enumerate(keys_and_waits):
        label = "backup" if key_idx else "primary"
        if key_idx:
            print("[API] switching to backup key")
        client = self._get_client(key)

        handled_400 = False
        for attempt, wait_sec in enumerate(waits):
            if wait_sec > 0:
                print(f"[API] {label} key attempt {attempt + 1}/{len(waits)}: waiting {wait_sec}s...")
                time.sleep(wait_sec)
            try:
                resp = client.chat.completions.create(**kwargs)
                if resp.choices:
                    return resp
                last_exc = Exception(
                    f"API returned choices=None ({label} key, attempt {attempt + 1})"
                )
                print(f"[API] choices=None from {label} key (attempt {attempt + 1})")
            except Exception as exc:
                if getattr(exc, "status_code", None) == 400:
                    handled_400 = True
                    messages = kwargs.get("messages", [])
                    rolled_back = False
                    for i in range(len(messages) - 1, -1, -1):
                        if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
                            del messages[i:]
                            rolled_back = True
                            break
                    if rolled_back:
                        print("[API] 400 — rolled back last assistant message, retrying once")
                        try:
                            resp = client.chat.completions.create(**kwargs)
                            if resp.choices:
                                return resp
                            last_exc = Exception("API returned choices=None after 400 rollback")
                        except Exception as retry_exc:
                            last_exc = retry_exc
                            print(f"[API] 400 rollback retry failed: {retry_exc}")
                    else:
                        last_exc = exc
                        print("[API] 400 — no assistant message to roll back, giving up")
                    break  # exit inner waits loop; handled_400 exits outer too
                else:
                    last_exc = exc
                    print(f"[API] error from {label} key (attempt {attempt + 1}): {exc}")
        if handled_400:
            break  # exit outer keys loop — 400 is terminal, don't try backup key

    raise last_exc or Exception("API: all retry attempts exhausted")
```

- [ ] **Step 1.4 — Run the new tests**

```bash
python -m pytest tests/test_resilient_client.py -k "400" -v
```

Expected: 4 PASSED

- [ ] **Step 1.5 — Run the full test suite to check for regressions**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests still PASS

- [ ] **Step 1.6 — Commit**

```bash
git add mission_executor/agent_main.py tests/test_resilient_client.py
git commit -m "fix: roll back last assistant message on 400 Bad Request in _ResilientClient"
```

---

## Task 2: `max_tokens=1000` for Agent and Critic

**Files:**
- Modify: `mission_executor/agent_main.py:280-287` (agent call)
- Modify: `mission_executor/critic.py:183-188` (critic call)

- [ ] **Step 2.1 — Add `max_tokens=1000` to agent call**

In `agent_main.py`, locate the `ao_client.chat.completions.create(...)` call in `agent_cycle` (~line 280). Add `max_tokens=1000`:

```python
res = ao_client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=traj,
    tools=tools,
    tool_choice="auto",
    stream=False,
    temperature=OPENAI_TEMPERATURE,
    max_tokens=1000,
)
```

- [ ] **Step 2.2 — Add `max_tokens=1000` to critic call**

In `critic.py`, locate `self._client.chat.completions.create(...)` in `ComplianceCritic.evaluate()` (~line 183). Add `max_tokens=1000`:

```python
resp = self._client.chat.completions.create(
    model=self._model,
    messages=messages,
    temperature=0.0,
    stream=False,
    max_tokens=1000,
)
```

- [ ] **Step 2.3 — Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all PASS (tests don't assert on kwargs, so no regression expected)

- [ ] **Step 2.4 — Commit**

```bash
git add mission_executor/agent_main.py mission_executor/critic.py
git commit -m "fix: cap agent and critic output at max_tokens=1000 to prevent token explosion"
```

---

## Task 3: Preserve Artifacts on CriticFailure

**Files:**
- Modify: `run.py:224-233`

- [ ] **Step 3.1 — Remove `shutil.rmtree` and update log/print strings**

In `run.py`, find the `if critic_aborted:` block (lines 225–233). The net change is: **remove line 232** (`shutil.rmtree(artifact_dir, ignore_errors=True)`) and update the `print` and `append_log` strings. The `try/except abort_reason` block already exists — don't touch it. Result:

```python
if critic_aborted:
    try:
        abort_reason = (artifact_dir / "critic_aborted").read_text(encoding="utf-8").strip()
    except Exception:
        abort_reason = "(reason unreadable)"
    append_log(run_log, f"CRITIC ABORTED: artifacts kept for inspection — {abort_reason}")
    print(f"[CRITIC ABORTED] {abort_reason} — keeping {artifact_dir}")
    # Still clean up containers below
```

The `shutil.rmtree` call is removed. All artifact files (`critic_aborted`, `result.json`, `messages_record.txt`, `critic_metrics.json`) remain on disk for post-hoc inspection.

- [ ] **Step 3.2 — Verify `shutil` import is still needed**

`shutil` is still used in `build_artifact_dir` (line 96: `shutil.rmtree(base, ignore_errors=True)`). No import change needed.

- [ ] **Step 3.3 — Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 3.4 — Commit**

```bash
git add run.py
git commit -m "fix: keep artifact folder on CriticFailure instead of deleting it"
```

---

## Task 4: Gemini 3.1 Models + Separate Critic Model

**Files:**
- Modify: `run_experiments.py:67-73` (ExperimentSetting dataclass)
- Modify: `run_experiments.py:60-64` (EXPERIMENT_SETTINGS)
- Modify: `run_experiments.py:159-168` (run_benchmarks_for_job cmd)
- Modify: `run_experiments.py:174-176` (job log lines)

- [ ] **Step 4.1 — Add `critic_model` field to `ExperimentSetting`**

Find the `ExperimentSetting` dataclass (~line 67):

```python
@dataclass(frozen=True)
class ExperimentSetting:
    base_url: str
    model_name: str
    result_folder_name: str
    critic_mode: str
```

Add `critic_model: str = ""` as the last field:

```python
@dataclass(frozen=True)
class ExperimentSetting:
    base_url: str
    model_name: str
    result_folder_name: str
    critic_mode: str
    critic_model: str = ""
```

The existing `settings = [ExperimentSetting(*row) for row in EXPERIMENT_SETTINGS]` on line 404 continues to work: 4-element tuples unpack to the first 4 fields; 5-element tuples will unpack to all 5.

- [ ] **Step 4.2 — Update `EXPERIMENT_SETTINGS` and its header comment**

Also update the comment on line 58 (`# List of experiment settings: (base_url, model_name, result_folder_name, critic_mode)`) to include `critic_model`.

Replace the current `EXPERIMENT_SETTINGS` list (~lines 60–64):

```python
EXPERIMENT_SETTINGS = [
    ("https://openrouter.ai/api/v1", "google/gemini-3.1-pro-preview", "gemini-31-pro-baseline", "none",  ""),
    ("https://openrouter.ai/api/v1", "google/gemini-3.1-pro-preview", "gemini-31-pro-blind",    "blind", "google/gemini-3.1-flash-lite-preview"),
    ("https://openrouter.ai/api/v1", "google/gemini-3.1-pro-preview", "gemini-31-pro-aware",    "aware", "google/gemini-3.1-flash-lite-preview"),
]
```

- [ ] **Step 4.3 — Wire `--critic-model` into `run_benchmarks_for_job` command**

Find the `cmd` list in `run_benchmarks_for_job` (~lines 159–168):

```python
cmd = [
    sys.executable,
    str(job.runtime_dir / "run_benchmarks.py"),
    "--openai-base-url",
    job.setting.base_url,
    "--openai-model",
    job.setting.model_name,
    "--critic-mode",
    job.setting.critic_mode,
]
```

Add `--critic-model` at the end:

```python
cmd = [
    sys.executable,
    str(job.runtime_dir / "run_benchmarks.py"),
    "--openai-base-url",
    job.setting.base_url,
    "--openai-model",
    job.setting.model_name,
    "--critic-mode",
    job.setting.critic_mode,
    "--critic-model",
    job.setting.critic_model,
]
```

- [ ] **Step 4.4 — Add critic model to job log**

Find the log lines in `run_benchmarks_for_job` (~line 174):

```python
log.write(f"Critic mode: {job.setting.critic_mode}\n")
```

Add a line after it:

```python
log.write(f"Critic mode: {job.setting.critic_mode}\n")
log.write(f"Critic model: {job.setting.critic_model or '(same as agent)'}\n")
```

- [ ] **Step 4.5 — Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 4.6 — Commit**

```bash
git add run_experiments.py
git commit -m "feat: add critic_model field to ExperimentSetting, switch to Gemini 3.1 models"
```

---

## Final Check

- [ ] Run full test suite one last time

```bash
python -m pytest tests/ -v
```

Expected: all PASS, no regressions.
