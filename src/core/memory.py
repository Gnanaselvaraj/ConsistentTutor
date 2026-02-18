"""
memory.py: Session memory, context, and summarization
"""
from typing import Any, List
import threading

class SessionMemory:
    def __init__(self):
        self.summary = None
        self.full_chat_history = []
        self.lock = threading.Lock()

    def add_message(self, user: str, message: str):
        with self.lock:
            self.full_chat_history.append({"user": user, "message": message})

    def get_history(self) -> List[Any]:
        with self.lock:
            return list(self.full_chat_history)

    def update_summary(self, llm, callback=None):
        def summarization_worker():
            summary = llm.invoke(f"Summarize: {self.full_chat_history}")
            self.summary = summary
            if callback:
                callback(summary)
        thread = threading.Thread(target=summarization_worker)
        thread.start()
        return None
