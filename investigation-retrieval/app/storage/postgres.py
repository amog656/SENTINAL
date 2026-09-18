from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import get_settings
from app.storage.models import Base

settings = get_settings()

connect_args = {}
if settings.postgres_url.startswith("postgresql"):
    connect_args["connect_timeout"] = 1

engine = create_engine(settings.postgres_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
