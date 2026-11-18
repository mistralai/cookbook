from env._base import BaseEnv


class Env(BaseEnv):
    workflows_base_url: str = "https://api.mistral.ai"
    deployment_name: str | None = None


env = Env()
