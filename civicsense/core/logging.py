"""Structured logging framework using Loguru and Rich.

Provides centralized log configuration with separate log files for
application, AI, database, errors, and audit events.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from civicsense.core.config import get_config

console = Console()

_initialized = False


def setup_logging() -> None:
    """Initialize the structured logging framework.

    Configures Loguru with separate sinks for application, AI, database,
    error, and audit logs. Enables rich tracebacks and structured output.
    """
    global _initialized
    if _initialized:
        return

    config = get_config()
    log_dir = Path(config.storage.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=config.logging.level,
        colorize=True,
        backtrace=config.logging.rich_tracebacks,
        diagnose=True,
    )

    logger.add(
        str(log_dir / "app.log"),
        format=log_format,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression=config.logging.compression,
        serialize=config.logging.json_output,
        encoding="utf-8",
    )

    logger.add(
        str(log_dir / "ai.log"),
        format=log_format,
        level="DEBUG",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression=config.logging.compression,
        serialize=config.logging.json_output,
        filter=lambda record: "ai" in record["extra"].get("module", ""),
        encoding="utf-8",
    )

    logger.add(
        str(log_dir / "database.log"),
        format=log_format,
        level="DEBUG",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression=config.logging.compression,
        serialize=config.logging.json_output,
        filter=lambda record: "database" in record["extra"].get("module", ""),
        encoding="utf-8",
    )

    logger.add(
        str(log_dir / "error.log"),
        format=log_format,
        level="ERROR",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression=config.logging.compression,
        serialize=config.logging.json_output,
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )

    logger.add(
        str(log_dir / "audit.log"),
        format=log_format,
        level="INFO",
        rotation=config.logging.rotation,
        retention=config.logging.retention,
        compression=config.logging.compression,
        serialize=config.logging.json_output,
        filter=lambda record: "audit" in record["extra"].get("module", ""),
        encoding="utf-8",
    )

    _initialized = True
    logger.info("Logging framework initialized", module="app")


def get_logger(module: str = "app") -> Any:
    """Return a context-bound logger for the given module.

    Args:
        module: Module identifier for log routing.

    Returns:
        A Loguru logger bound to the specified module context.
    """
    return logger.bind(module=module)
