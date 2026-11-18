from env._base import BaseEnv


class Env(BaseEnv):
    mistral_api_key: str | None = None
    # Optional Mistral REST API base URL. None uses the SDK default.
    mistral_base_url: str | None = None


env = Env()
