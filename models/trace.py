from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from database import Base

class Trace(Base):
    __tablename__ = "traces"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    thread_id = Column(String, nullable=False)
    type = Column(String, nullable=False)
    msg_id = Column(String)
    content = Column(String)
    tool_call_requests = Column(JSONB)
    tool_name = Column(String)
    tool_call_id = Column(String)
    run_start_time = Column(DateTime, nullable=False)
    run_end_time = Column(DateTime)