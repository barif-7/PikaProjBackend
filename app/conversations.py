from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Optional


_STORE_LOCK = Lock()


@dataclass(frozen=True)
class ConversationStore:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._conversations_path.touch(exist_ok=True)

    @property
    def _conversations_path(self) -> Path:
        return self.root_dir / "conversations.json"

    def fetch(self, *, user_id: str, conversation_id: str = "default") -> dict[str, Any]:
        with _STORE_LOCK:
            conversations = _load_json_map(self._conversations_path)

        return (
            conversations
            .get(user_id, {})
            .get(conversation_id, _default_conversation_payload(conversation_id))
        )

    def save(
        self,
        *,
        user_id: str,
        conversation_id: str,
        summary: Optional[str],
        voice_profile_id: Optional[str],
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        payload = {
            "conversation_id": conversation_id,
            "summary": summary or "",
            "voice_profile_id": voice_profile_id,
            "messages": messages,
        }

        with _STORE_LOCK:
            conversations = _load_json_map(self._conversations_path)
            user_conversations = conversations.get(user_id, {})
            user_conversations[conversation_id] = payload
            conversations[user_id] = user_conversations
            _save_json_map(self._conversations_path, conversations)

        return payload


def make_conversation_store(root_dir: str) -> ConversationStore:
    return ConversationStore(root_dir=Path(root_dir))


def _load_json_map(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_json_map(path: Path, payload: dict[str, Any]) -> None:
    # Atomic write to avoid leaving a half-written file if the process dies
    # mid-save; a corrupt conversations.json would erase chat history.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _default_conversation_payload(conversation_id: str) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "summary": "",
        "voice_profile_id": None,
        "messages": [],
    }
