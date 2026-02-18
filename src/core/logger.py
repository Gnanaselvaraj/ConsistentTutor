"""
logger.py: Logging and versioning
"""
import os
import json
import datetime
import hashlib

class ConversationLogger:
    def __init__(self, log_dir: str = "logs", app_version: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if app_version is None:
            try:
                with open(__file__, 'rb') as f:
                    code_hash = hashlib.md5(f.read()).hexdigest()[:8]
                self.app_version = f"auto-{code_hash}"
            except Exception:
                self.app_version = "auto-unknown"
        else:
            self.app_version = app_version
        self.session_log = []

    def log(self, user_input, system_output):
        entry = {
            "version": self.app_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_input": user_input,
            "system_output": system_output
        }
        self.session_log.append(entry)
        with open(os.path.join(self.log_dir, "session_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
