from api.health.metadata.deployment import metadata
from env.workflows import env as workflows_env


def test_deployment_metadata_reads_the_workflows_environment() -> None:
    assert metadata() == workflows_env.deployment_name
