from datetime import datetime
from database import SessionLocal
from models.trace import Trace

class TraceManager:
    def __init__(self) -> None:
        pass

    def add_trace(self, thread_id: str, agent_response: dict, start_time: datetime, end_time: datetime) -> None:
        traces = self.get_traces_from_messages(thread_id, agent_response["messages"], start_time, end_time)

        with SessionLocal() as db:
            db.add_all(traces)
            db.commit()

    def get_traces_from_messages(self, thread_id: str, messages: list, start_time: datetime, end_time: datetime) -> list:
        traces = []

        for msg in messages:
            type = msg.type

            traces.append(
                Trace(
                    id=msg.id,
                    thread_id=thread_id,
                    type=type,
                    content=msg.content if len(msg.content) else None,
                    tool_call_requests=[ {"name": request["name"], 'args': request["args"]} for request in msg.tool_calls] if type == "ai" and hasattr(msg, "tool_calls") else None,
                    tool_name=msg.name if type == "tool" and hasattr(msg, "tool_call_id") else None,
                    tool_call_id=msg.tool_call_id if hasattr(msg, "tool_call_id") else None,
                    run_start_time=start_time,
                    run_end_time=end_time
                )
            )


        return traces