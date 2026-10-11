"""Shared endpoint factory tests."""
from types import SimpleNamespace
from podclipper.llm import build_provider


def test_build_provider_uses_shared_endpoint(mocker):
    fake = mocker.patch("podclipper.llm.LiteLLMProvider")
    cfg = SimpleNamespace(endpoint="https://example.test/v1", model="example")
    assert build_provider(cfg) is fake.return_value
    fake.assert_called_once_with(cfg, model="example")
