from typing import Literal

from env._base import BaseEnv


class Env(BaseEnv):
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"


env = Env()
