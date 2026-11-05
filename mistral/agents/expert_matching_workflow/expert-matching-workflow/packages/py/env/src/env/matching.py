"""Settings for expert matching: the models that read a profile and pick a role."""

from env._base import BaseEnv


class Env(BaseEnv):
    ocr_model: str = "mistral-ocr-latest"
    matching_model: str = "mistral-medium-latest"


env = Env()
