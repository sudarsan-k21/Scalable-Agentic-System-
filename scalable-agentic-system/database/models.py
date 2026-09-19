"""
Durable state, separate from the Tool Registry (router/tool_registry.py)
and separate from Redis (which holds ephemeral session/working state).

- execution_log: append-only audit trail of every tool call — powers
  System Search Tool's `log_search`, observability dashboards, and the
  `historical_success_rate` feedback loop into router/ranking.py.
- sessions: durable conversation summaries, so a session can resume even
  if Redis is cold (cache miss / restart).
"""
from __future__ import annotations
from sqlalchemy import (
    Column, String, Boolean, Text, Integer, TIMESTAMP, func, create_engine
)
from sqlalchemy.orm import declarative_base
from app.config import settings

Base = declarative_base()


class ExecutionLog(Base):
    __tablename__ = "execution_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    tool_id = Column(String, index=True, nullable=False)
    args = Column(Text)              # JSON-serialized
    success = Column(Boolean, nullable=False)
    output = Column(Text)            # JSON-serialized
    error = Column(Text, nullable=True)
    retries = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    conversation_summary = Column(Text, default="")
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


def get_engine():
    return create_engine(settings.postgres_url)


def init_models():
    Base.metadata.create_all(get_engine())
