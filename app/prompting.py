from __future__ import annotations

from .models import ChatHistoryMessage
from typing import Dict, List


SYSTEM_PROMPT = (
    "You are SEMI, the user's AI self. Speak like a grounded, emotionally intelligent "
    "future version of the user. Be concise, warm, and direct. Keep responses under "
    "three sentences, suitable for spoken playback. Preserve continuity with the prior "
    "conversation. Do not mention being an AI model unless asked directly."
)


def build_chat_messages(
    history: List[ChatHistoryMessage],
    transcript: str,
    conversation_summary: str | None = None,
    max_history_messages: int = 8,
) -> List[Dict[str, str]]:
    trimmed_history = history[-max_history_messages:]
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    summary = (conversation_summary or "").strip()
    if summary:
        messages.append(
            {
                "role": "system",
                "content": "Conversation summary so far:\n" + summary,
            }
        )

    for item in trimmed_history:
        role = "assistant" if item.role == "assistant" else "user"
        content = item.content.strip()
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": transcript.strip()})
    return messages
