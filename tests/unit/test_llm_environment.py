"""User-visible CLI environment configuration; no network calls."""

import yaml

from podclipper.config import load_default_config
from podclipper.main import main


def test_dotenv_endpoint_overrides_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_ENDPOINT", raising=False)
    (tmp_path / "podclipper.yaml").write_text(
        "llm:\n  endpoint: https://yaml.example/v1\n"
    )
    (tmp_path / ".env").write_text("LLM_ENDPOINT=https://env.example/v1\n")

    assert main(["config", "--resolved"]) == 0

    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["llm"]["endpoint"] == "https://env.example/v1"


def test_dotenv_model_overrides_yaml(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    (tmp_path / "podclipper.yaml").write_text("llm:\n  model: yaml-model\n")
    (tmp_path / ".env").write_text("LLM_MODEL=env-model\n")

    assert main(["config", "--resolved"]) == 0

    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["llm"]["model"] == "env-model"


def test_init_creates_one_generic_key_setting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert main(["init"]) == 0

    config = yaml.safe_load((tmp_path / "podclipper.yaml").read_text())
    assert "api_key_env" not in config["llm"]
    env = (tmp_path / ".env").read_text()
    assert "\nLLM_API_KEY=\n" in env
    assert "OPENAI_API_KEY" not in env
    assert "OPENROUTER_API_KEY" not in env


def test_setup_accepts_the_generic_api_key(monkeypatch):
    from podclipper.main import _check_setup

    monkeypatch.setenv("LLM_API_KEY", "generic-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("podclipper.main.shutil.which", lambda name: "/usr/bin/" + name)

    _check_setup(load_default_config())


def test_resolved_config_has_no_key_variable_setting(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert main(["config", "--resolved"]) == 0

    resolved = yaml.safe_load(capsys.readouterr().out)
    assert "api_key_env" not in resolved["llm"]
