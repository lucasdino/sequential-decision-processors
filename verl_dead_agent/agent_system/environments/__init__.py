__all__ = ["EnvironmentManagerBase", "make_envs"]

def __getattr__(name):
    if name in __all__:
        from .env_manager import EnvironmentManagerBase, make_envs
        return {"EnvironmentManagerBase": EnvironmentManagerBase, "make_envs": make_envs}[name]
    raise AttributeError(name)