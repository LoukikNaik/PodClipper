# TDD Plan — Phase 1 (LiteLLM swap) + Phase 2 (PyPI package)

Branch: `feat/litellm-and-pypi-package`
Discipline: strict TDD via the `tdd` skill (RED → verify-RED → GREEN → REFACTOR).
Safety net: 187 Phase-0 characterization tests — must stay green throughout.
Commit pattern: one commit per successful GREEN (or GREEN + REFACTOR).

---

## Phase 1 — LiteLLM as the unified API provider

**Done state:**
- New `src/llm/litellm_provider.py` implementing `LLMProvider` Protocol.
- `build_provider()` dispatches `"litellm"` → `LiteLLMProvider`.
- `src/llm/anthropic_api.py` deleted; its branch removed from factory.
- `--llm-provider` choices: `{claude_cli, litellm}`.
- `config/default.yaml`: `llm.litellm` section added; `llm.anthropic_api` section removed.
- `claude_cli` provider unchanged.
- All 187 Phase-0 tests still pass (with the 3 time-bombed ones updated, 1 file deleted).

**Plumbing (pre-cycle, not TDD):**
- ✅ `pip install litellm` and add `litellm>=1.50` to `requirements.txt`.

**TDD cycles** (one new test file: `tests/unit/test_llm_litellm.py`):

| # | RED (test name) | What GREEN adds |
|---|---|---|
| 1.1 | `test_init_sets_name_attribute_to_litellm_and_stores_model` | Create `src/llm/litellm_provider.py` with `LiteLLMProvider(cfg, model)` storing model + `name = "litellm"`. |
| 1.2 | `test_complete_returns_message_content_from_litellm_response` | Implement `complete()` calling `litellm.completion(...)` and returning `response.choices[0].message.content`. Use `mocker.patch("litellm.completion")`. |
| 1.3 | `test_complete_passes_configured_model_string_to_litellm` | Forward `self.model` as `model=` kwarg. |
| 1.4 | `test_complete_wraps_user_prompt_in_user_role_message` | Build `messages=[{"role":"user","content":user_prompt}]`. |
| 1.5 | `test_complete_prepends_system_prompt_as_system_message_when_provided` | Conditional prepend `{"role":"system","content":system_prompt}` when non-empty. |
| 1.6 | `test_complete_forwards_max_tokens_kwarg_to_litellm` | Pass `max_tokens=` kwarg. |
| 1.7 | `test_complete_raises_llmerror_when_litellm_call_raises` | Wrap `litellm.completion(...)` in try/except, raise `LLMError`. |
| 1.8 | `test_complete_raises_llmerror_when_response_content_is_empty` | Check `not text.strip()` after extraction; raise `LLMError`. |
| 1.9 | `test_complete_forwards_api_base_when_set_in_cfg` | Conditional `api_base=cfg.api_base` kwarg (for Ollama/vLLM/proxies). |
| 1.10 | `test_complete_forwards_timeout_seconds_from_cfg` | Pass `timeout=cfg.timeout_seconds` kwarg. |
| 1.11 | `test_complete_forwards_num_retries_from_cfg` | Pass `num_retries=cfg.num_retries` kwarg. |
| 1.12 | `test_build_provider_routes_litellm_to_litellmprovider` (in `test_llm_factory.py`) | Add branch to `build_provider()` for `"litellm"`. |

**Refactor enabled by the safety net** (not new TDD cycles — characterization tests are the proof):

| # | Action | Tests touched |
|---|---|---|
| 1.13 | Update `--llm-provider` choices: `{claude_cli, anthropic_api}` → `{claude_cli, litellm}` in `main.py:build_parser()`. | Rewrite `test_parser_llm_provider_choices_are_claude_cli_and_anthropic_api` → `_and_litellm` in `test_main.py`. Rewrite `test_apply_cli_overrides_sets_llm_provider` to use `"litellm"` as the value (was `"anthropic_api"`). |
| 1.14 | Delete `src/llm/anthropic_api.py` + the `anthropic_api` branch in `src/llm/__init__.py:build_provider()`. | Delete entire `tests/unit/test_llm_anthropic_api.py` (14 tests). Delete `test_build_provider_routes_anthropic_api_to_anthropicapiprovider` from `test_llm_factory.py`. |
| 1.15 | Update `config/default.yaml`: remove `llm.anthropic_api` section, add `llm.litellm` section with `api_key_env`, `api_base: null`, `timeout_seconds`, `num_retries`. | Possibly extend `test_load_config_loads_real_default_yaml_*` to assert `litellm` subsection exists. |

**Manual smoke after 1.15:** run `python main.py <video> --llm-provider litellm` with `llm.model: "anthropic/claude-sonnet-4-5"` and confirm parity with `--llm-provider claude_cli`. NOT a unit test — real API call + real pipeline.

**Open questions** (decide before starting):
- **Q1.A:** Keep `anthropic` in `requirements.txt` after we delete `anthropic_api.py`? LiteLLM uses it as a lazy dep when routing to `anthropic/...` models. **Default decision:** keep, since it's small and LiteLLM needs it for Anthropic routing.
- **Q1.B:** Default `llm.provider` in `config/default.yaml` — stay on `claude_cli` (current default) or flip to `litellm`? **Default decision:** stay on `claude_cli` — zero behavior change for existing users.
- **Q1.C:** Default `llm.model` for the litellm section — `"anthropic/claude-sonnet-4-5"` to mirror current behavior, or leave commented-out so user must explicitly choose? **Default decision:** set `"anthropic/claude-sonnet-4-5"` so users get something working out of the box.

---

## Phase 2 — PyPI package conversion (pip + uv)

**Done state:**
- `src/` renamed to `src/podclipper/`; all `from src.X import …` → `from podclipper.X import …`.
- `pyproject.toml` (PEP 621) declares package metadata, deps, console entry point.
- Python `>=3.10,<3.13`.
- `config/default.yaml` and `prompts/*.txt` bundled as package data via `importlib.resources`.
- One console script: `podclipper = "podclipper.main:main"`.
- `regen_crops.py`, `debug_detect_clip.py` moved to `dev/` (not installed).
- Heavy deps (`pyannote.audio`, `torch`) become `[diarize]` extra.
- GitHub Action `.github/workflows/publish-pypi.yml` publishes on tag push.
- All 187 Phase-0 tests + 12 Phase-1 cycles still pass after rename.

**Plumbing (pre-cycle, not TDD):**
- Verify `pyproject.toml` doesn't already exist (won't — Phase 0 confirmed).
- Snapshot of all `from src.` import sites for the rename pass.

**TDD cycles for the testable parts:**

| # | RED (test name) | What GREEN adds |
|---|---|---|
| 2.1 | `test_package_version_is_importable_from_podclipper` | Move `src/__init__.py` to `src/podclipper/__init__.py` with `__version__ = "0.1.0"`. Update `pythonpath` in `pytest.ini` if needed; update one test's import path. |
| 2.2 | `test_load_default_config_returns_simplenamespace_via_importlib_resources` | Add `podclipper.config.load_default_config()` that uses `importlib.resources.files("podclipper").joinpath("data/default.yaml").read_text()`. Move `config/default.yaml` into `src/podclipper/data/default.yaml` (or whatever importlib.resources path we choose). |
| 2.3 | `test_load_prompt_returns_file_contents_via_importlib_resources` | Add `podclipper.prompts.load_prompt(name)` similarly. Move `prompts/*.txt` into `src/podclipper/prompts/`. |
| 2.4 | `test_analyze_uses_packaged_prompt_path_not_repo_relative_path` | Update `src/podclipper/analyze.py` to use `podclipper.prompts.load_prompt("reel_detector.txt")` instead of `Path(__file__).parent.parent / "prompts" / ...`. Same for `trailer.py`, `evaluate.py`. |
| 2.5 | `test_console_entry_point_podclipper_prints_help_and_exits_zero` | Add `[project.scripts] podclipper = "podclipper.main:main"` to `pyproject.toml`. Test via `subprocess.run(["podclipper", "--help"])`. Requires `pip install -e .` as a fixture setup. |

**Refactor enabled by the safety net:**

| # | Action | Risk |
|---|---|---|
| 2.6 | Mass rename `src/` → `src/podclipper/`. Update all `from src.X` → `from podclipper.X` in production code AND tests (~30 import lines). | Phase-0 + Phase-1 tests catch any miss. |
| 2.7 | Write `pyproject.toml` (PEP 621): metadata, deps from `requirements.txt`, `[project.optional-dependencies] diarize`, console script. | Run `pip install -e .` in a fresh venv as smoke test. |
| 2.8 | Move `regen_crops.py`, `debug_detect_clip.py` from repo root to `dev/`. Update their `sys.path.insert(0, str(...))` and import statements if needed. | Manual run of each to confirm still works. |
| 2.9 | Add `.github/workflows/publish-pypi.yml` (using `pypa/gh-action-pypi-publish`). | Manual: dry-run via `python -m build` locally. |

**Open questions** (decide before starting Phase 2):
- **Q2.A:** Where do bundled `config/` and `prompts/` live inside the package? Options: `src/podclipper/data/{default.yaml,prompts/*.txt}` or two siblings `src/podclipper/{config,prompts}/`. **Default decision:** sibling directories for clarity — `src/podclipper/prompts/*.txt` and `src/podclipper/config/default.yaml`.
- **Q2.B:** Should `config/default.yaml` at the repo ROOT stay (for dev workflows where users want a writable default to override) or be deleted (single source of truth in package)? **Default decision:** delete the repo-root copy; the in-package one is the canonical default; users override via `-c /path/to/their.yaml`.
- **Q2.C:** Package name on PyPI: `podclipper`. Check availability before committing. **Action:** quick `curl https://pypi.org/pypi/podclipper/json` before Phase 2 starts.
- **Q2.D:** Should the `landing/` directory be included in the package? **Default decision:** no — it's the website, ships separately via GH Pages.
- **Q2.E:** Do we add `[diarize]` extra or just delete `pyannote`/`torch` from required deps? **Default decision:** add as extra since the deprecated path still references it — `pip install podclipper[diarize]` works, plain `pip install podclipper` skips the 5GB torch install.

---

## Total estimated work

| Phase | Cycles | Net new files | Net deleted files | Estimated session time |
|---|---|---|---|---|
| 1 | 12 TDD + 3 refactor | 1 (`test_llm_litellm.py` + `litellm_provider.py`) | 1 (`anthropic_api.py` + `test_llm_anthropic_api.py`) | 1-2 hrs |
| 2 | 5 TDD + 4 refactor | `pyproject.toml`, `dev/`, packaged data layout | `src/__init__.py`, repo-root `config/`, repo-root `prompts/`, repo-root debug scripts | 2-3 hrs |

After both phases: rebase or merge feature branch back to main, then tag + push to publish.

---

## Notes for the next session

- Every cycle must show the failing test output before the GREEN edit. No "trust me it would fail."
- After each cycle: run the **full** suite (`pytest tests/unit/`), not just the new test.
- If a Phase-0 characterization test breaks during Phase 1, treat it as Phase 1 having a real regression (likely) OR as an intentional contract change that needs to be acknowledged in the commit (rare — only for the 3 time-bombed tests in `test_llm_*` and `test_main.py`).
- The "fail loudly" assertion is real: the 3 time-bombed tests are the only Phase-0 tests we expect to need updating in Phase 1. If a different test fails, stop and investigate.
- For Phase 2, the rename is the highest-risk change. Do it on a clean working tree, run the suite immediately after, revert immediately if anything goes red — don't try to fix forward.
