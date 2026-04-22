from app.models import ChatHistoryMessage
from app.prompting import SYSTEM_PROMPT, build_chat_messages


def test_build_chat_messages_keeps_system_and_latest_history() -> None:
    history = [
        ChatHistoryMessage(role="user", content=f"user-{index}") if index % 2 == 0
        else ChatHistoryMessage(role="assistant", content=f"assistant-{index}")
        for index in range(10)
    ]

    messages = build_chat_messages(history, "latest transcript", max_history_messages=4)

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1:] == [
        {"role": "assistant", "content": "assistant-6"},
        {"role": "user", "content": "user-7"},
        {"role": "assistant", "content": "assistant-8"},
        {"role": "user", "content": "user-9"},
        {"role": "user", "content": "latest transcript"},
    ]


def test_build_chat_messages_strips_empty_content() -> None:
    history = [
        ChatHistoryMessage(role="user", content="  "),
        ChatHistoryMessage(role="assistant", content="  grounded reply  "),
    ]

    messages = build_chat_messages(history, "  final turn  ", max_history_messages=8)

    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "grounded reply"},
        {"role": "user", "content": "final turn"},
    ]
