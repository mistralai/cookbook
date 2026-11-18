"""Settings for optional OTLP log export.

Export uses the configured application credential and remains inert when no credential is present.
"""

from env._base import BaseEnv


class Env(BaseEnv):
    # Public OTLP ingestion endpoint; override for another compatible collector.
    telemetry_endpoint: str = "https://api.mistral.ai/telemetry/v1/logs"

    # Name shown for this exporting service.
    telemetry_service_name: str = "app-workspace"

    # Set false to disable export for the application.
    telemetry_enabled: bool = True

    # Bound shutdown time so queued telemetry cannot delay application termination indefinitely.
    telemetry_export_timeout_seconds: float = 5.0


env = Env()
