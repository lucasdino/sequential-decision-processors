# Mainly modeling this based off the file structure of how they implemented the alfworld agent
def get_environment(env_type):
    if env_type == 'LifegateTWEnv':
        from agent_system.environments.env_package.lifegate.lifegate.agents.environment.lifegate_env import LifegateTWEnv
        return LifegateTWEnv
    else:
        raise NotImplementedError(f"Environment {env_type} is not implemented.")
