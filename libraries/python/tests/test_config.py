"""Tests for pandaprobe.config."""

import pytest

from pandaprobe.config import SdkConfig, resolve_config


class TestSdkConfig:
    def test_raises_when_api_key_missing(self):
        with pytest.raises(ValueError, match="API key is required"):
            SdkConfig(project_name="proj")

    def test_raises_when_project_name_missing(self):
        with pytest.raises(ValueError, match="project name is required"):
            SdkConfig(api_key="sk_pp_abc")

    def test_no_error_when_disabled(self):
        cfg = SdkConfig(enabled=False)
        assert cfg.enabled is False
        assert cfg.api_key is None

    def test_defaults(self):
        cfg = SdkConfig(api_key="sk_pp_abc", project_name="proj")
        assert cfg.endpoint == "https://api.pandaprobe.com"
        assert cfg.batch_size == 10
        assert cfg.flush_interval == 5.0
        assert cfg.max_queue_size == 1000
        assert cfg.debug is False
        assert cfg.environment is None
        assert cfg.release is None


class TestResolveConfig:
    def test_constructor_overrides_env(self, monkeypatch):
        monkeypatch.setenv("PANDAPROBE_API_KEY", "env_key")
        monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "env_proj")
        cfg = resolve_config(api_key="override_key", project_name="override_proj")
        assert cfg.api_key == "override_key"
        assert cfg.project_name == "override_proj"

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("PANDAPROBE_API_KEY", "sk_pp_env")
        monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "my-env-proj")
        monkeypatch.setenv("PANDAPROBE_ENDPOINT", "http://custom:9000")
        monkeypatch.setenv("PANDAPROBE_BATCH_SIZE", "20")
        monkeypatch.setenv("PANDAPROBE_FLUSH_INTERVAL", "2.5")
        monkeypatch.setenv("PANDAPROBE_ENABLED", "true")
        monkeypatch.setenv("PANDAPROBE_DEBUG", "false")
        cfg = resolve_config()
        assert cfg.api_key == "sk_pp_env"
        assert cfg.project_name == "my-env-proj"
        assert cfg.endpoint == "http://custom:9000"
        assert cfg.batch_size == 20
        assert cfg.flush_interval == 2.5

    def test_enabled_false_from_env(self, monkeypatch):
        monkeypatch.setenv("PANDAPROBE_ENABLED", "false")
        cfg = resolve_config()
        assert cfg.enabled is False

    def test_optional_env_defaults(self, monkeypatch):
        monkeypatch.setenv("PANDAPROBE_API_KEY", "sk_pp_x")
        monkeypatch.setenv("PANDAPROBE_PROJECT_NAME", "p")
        cfg = resolve_config()
        assert cfg.environment is None
        assert cfg.release is None
        assert cfg.max_queue_size == 1000
