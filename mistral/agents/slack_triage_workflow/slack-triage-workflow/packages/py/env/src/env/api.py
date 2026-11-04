"""Who may call the API host from a browser.

Shipped by `fastapi`, not `core`: these settings describe the HTTP host, and nothing outside
`apps/api` reads them.
"""

from env._base import BaseEnv


class Env(BaseEnv):
    cors_origin: str = "http://localhost:3001"


env = Env()
