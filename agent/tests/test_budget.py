from agent.budget import count_tokens, total_tokens, trim_to_budget


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def test_count_tokens_nonzero() -> None:
    assert count_tokens("hello world") >= 1


def test_trim_keeps_system_and_last() -> None:
    messages = [
        _msg("system", "you are a helpful agent"),
        _msg("user", "old turn " * 200),
        _msg("assistant", "old reply " * 200),
        _msg("user", "the current question"),
    ]
    budget = total_tokens(messages) // 2
    trimmed = trim_to_budget(messages, budget)

    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "the current question"
    assert total_tokens(trimmed) <= total_tokens(messages)


def test_trim_preserves_last_even_when_over_budget() -> None:
    messages = [_msg("user", "x " * 1000)]
    trimmed = trim_to_budget(messages, budget=1)
    assert trimmed == messages


def test_trim_empty() -> None:
    assert trim_to_budget([], budget=100) == []


def test_trim_drops_oldest_first() -> None:
    messages = [
        _msg("system", "sys"),
        _msg("user", "AAAA " * 50),
        _msg("user", "BBBB " * 50),
        _msg("user", "current"),
    ]
    # Budget enough for system + one middle message + last, not both middles.
    budget = total_tokens(messages) - total_tokens([messages[1]])
    trimmed = trim_to_budget(messages, budget)
    contents = [m["content"] for m in trimmed]
    assert "current" in contents
    assert contents[0] == "sys"
    # The oldest middle (AAAA) is dropped before the newer (BBBB).
    assert not any(c.startswith("AAAA") for c in contents)
