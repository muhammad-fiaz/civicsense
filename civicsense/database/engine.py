"""SQLAlchemy database engine and session factory.

Provides centralized engine creation, session management, and
database initialization utilities for CivicSense.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from civicsense.core.config import get_config
from civicsense.core.logging import get_logger
from civicsense.database.models.orm import Base

logger = get_logger("database")

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the global SQLAlchemy engine singleton.

    Returns:
        The configured SQLAlchemy engine.
    """
    global _engine
    if _engine is None:
        config = get_config()
        _engine = create_engine(
            config.database.url,
            echo=config.database.echo,
            pool_pre_ping=True,
        )

        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(
            dbapi_connection: object, connection_record: object
        ) -> None:
            """Enable WAL mode and foreign keys for SQLite connections."""
            if config.database.url.startswith("sqlite"):
                cursor = dbapi_connection.cursor()  # type: ignore[union-attr]
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info("Database engine created", module="database")
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the global session factory singleton.

    Returns:
        The configured sessionmaker.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Provide a transactional database session.

    Yields:
        A SQLAlchemy Session instance.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database() -> None:
    """Create all database tables if they do not exist."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized", module="database")


def dispose_engine() -> None:
    """Dispose of the global database engine and reset state."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None
    logger.info("Database engine disposed", module="database")
