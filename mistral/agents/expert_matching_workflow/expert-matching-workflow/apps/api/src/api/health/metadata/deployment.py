from env.workflows import env as workflows_env


# The module's basename is the metadata key: the health route reads `deployment`, so renaming this
# file silently empties `HealthResponse.deployment`. Called once at discovery, so the value is read
# at boot.
def metadata() -> str | None:
    return workflows_env.deployment_name
