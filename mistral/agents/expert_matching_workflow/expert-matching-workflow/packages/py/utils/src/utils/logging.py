"""Shared structlog configuration for the generated application."""

import logging
import logging.config

import structlog


def configure_logging(*, log_level: str = "INFO", log_format: str = "console") -> None:
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if log_format == "json" else structlog.dev.ConsoleRenderer()
    )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        renderer,
                    ],
                    "foreign_pre_chain": shared_processors,
                },
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "default"},
            },
            "loggers": {
                "": {"handlers": ["console"], "level": log_level},
                # Not no-ops. Launchers apply uvicorn's LOGGING_CONFIG before importing the
                # app, which gives these loggers their own handler and `propagate: False` —
                # two formats from one process. Re-declaring them clears those handlers,
                # because dictConfig clears the handler list of every logger it configures,
                # and routes the records to root, so this pipeline is the only formatter. The
                # alternative is not available through every supported launcher.
                "uvicorn": {"handlers": [], "propagate": True},
                "uvicorn.error": {"handlers": [], "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": True},
            },
        }
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
