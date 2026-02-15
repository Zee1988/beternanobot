from nanobot.config.schema import Config


def test_queue_defaults_loaded():
    cfg = Config()
    assert cfg.agents.defaults.queue.mode == "followup"
    assert cfg.agents.defaults.auto_continue.enabled is True
