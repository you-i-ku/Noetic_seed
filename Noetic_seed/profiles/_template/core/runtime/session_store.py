"""Session Persistence — claw-code 準拠。

claw-code 参照: rust/crates/runtime/src/session.rs + session_store.py

責務: Session を JSON で保存/復元。ディレクトリ単位で履歴を管理し、
resume workflow (latest / id 指定) をサポート。
"""
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from core.runtime.session import Session


class SessionStore:
    """~/.claw/sessions/ 等のディレクトリに session を保存する。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session,
             session_id: Optional[str] = None,
             metadata: Optional[dict] = None) -> str:
        """session を保存。戻り値: session_id。"""
        sid = session_id or f"sess_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        path = self.root / f"{sid}.json"
        data = {
            "id": sid,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metadata": metadata or {},
            "messages": session.messages,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return sid

    def load(self, session_id: str) -> Optional[Session]:
        path = self.root / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        s = Session()
        s.messages = list(data.get("messages") or [])
        return s

    def load_latest(self) -> Optional[Session]:
        """最新の session を返す。"""
        files = sorted(self.root.glob("sess_*.json"))
        if not files:
            return None
        return self.load(files[-1].stem)

    def list_sessions(self, limit: int = 50) -> list:
        """最近の session メタ情報リスト (新しい順)。"""
        files = sorted(self.root.glob("sess_*.json"), reverse=True)
        out: list = []
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "id": data.get("id", f.stem),
                "saved_at": data.get("saved_at", ""),
                "message_count": len(data.get("messages") or []),
                "metadata": data.get("metadata") or {},
            })
        return out

    def delete(self, session_id: str) -> bool:
        path = self.root / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
