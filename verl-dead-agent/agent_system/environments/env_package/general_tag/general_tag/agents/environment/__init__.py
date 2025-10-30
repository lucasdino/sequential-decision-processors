# Mainly modeling this based off the file structure of how they implemented the alfworld agent
def get_environment(env_type):
    from agent_system.environments.env_package.general_tag.general_tag.agents.environment.general_env import GeneralTWEnv
    return GeneralTWEnv

