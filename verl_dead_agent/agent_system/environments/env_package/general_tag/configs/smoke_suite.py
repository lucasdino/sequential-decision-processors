"""Preset TALES environment settings for the env_manager smoke test.

Base config is in build_min_config() in env_manager.py.
These are only the surgical overrides needed per environment.
"""

SMOKE_MULTI_SUITE = {
    "train": [
        {"env_name": "tales_twx", "label": "tales_twx/train"},
        {"env_name": "tales_alfworld", "label": "tales_alfworld/train"},
    ],
    "val": [
        {"env_name": "tales_twx", "label": "tales_twx/val"},
        {
            "env_name": "tales_alfworld",
            "label": "tales_alfworld/val_seen",
            "overrides": {"env": {"valid_seen": True}},
        },
        {
            "env_name": "tales_alfworld",
            "label": "tales_alfworld/val_unseen",
            "overrides": {"env": {"valid_seen": False}},
        },
    ],
}
