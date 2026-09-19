"""Connection helpers for Postgres (durable) and Redis (fast/session cache)."""
from __future__ import annotations
import json
import redis
from sqlalchemy.orm import sessionmaker
from database.models import get_engine

SessionLocal = sessionmaker(bind=get_engine())
_redis = redis.from_url(__import__("app.config", fromlist=["settings"]).settings.redis_url)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def cache_get(key: str) -> dict | None:
    raw = _redis.get(key)
    return json.loads(raw) if raw else None


def cache_set(key: str, value: dict, ttl_seconds: int = 3600) -> None:
    _redis.set(key, json.dumps(value), ex=ttl_seconds)
