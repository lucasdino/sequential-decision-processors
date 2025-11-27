"""Preset TALES environment settings for the env_manager smoke test."""

SMOKE_MULTI_SUITE = {
    "train": [
        {
            "env_name": "tales_twx",
            "label": "tales_twx/train",
            "max_steps": 40,
            "overrides": {
                "env": {
                    "load_env_seeds": True,
                    "max_steps": 40,
                }
            },
        },
        {
            "env_name": "tales_alfworld",
            "label": "tales_alfworld/train",
            "max_steps": 25,
            "overrides": {
                "env": {
                    "load_env_seeds": True,
                    "max_steps": 25,
                }
            },
        },
    ],
    "val": [
        {
            "env_name": "tales_twx",
            "label": "tales_twx/val",
            "max_steps": 40,
            "overrides": {
                "env": {
                    "load_env_seeds": True,
                    "max_steps": 40,
                }
            },
        },
        {
            "env_name": "tales_alfworld",
            "label": "tales_alfworld/val_seen",
            "max_steps": 25,
            "overrides": {
                "env": {
                    "load_env_seeds": True,
                    "max_steps": 25,
                    "valid_seen": True,
                }
            },
        },
        {
            "env_name": "tales_alfworld",
            "label": "tales_alfworld/val_unseen",
            "max_steps": 25,
            "overrides": {
                "env": {
                    "load_env_seeds": True,
                    "max_steps": 25,
                    "valid_seen": False,
                }
            },
        },
    ],
}
