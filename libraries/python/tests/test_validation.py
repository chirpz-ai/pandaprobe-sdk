"""Tests for pandaprobe.validation."""

import pytest

from pandaprobe.validation import (
    extract_last_assistant_message,
    extract_last_user_message,
    validate_messages_format,
)


class TestValidateMessagesFormat:
    def test_valid_data(self):
        validate_messages_format(
            {"messages": [{"role": "user", "content": "hello"}]},
            "test",
        )

    def test_none_passes(self):
        validate_messages_format(None, "test")

    def test_content_none_passes(self):
        validate_messages_format(
            {"messages": [{"role": "assistant", "content": None}]},
            "test",
        )

    def test_content_list_passes(self):
        validate_messages_format(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "hello", "annotations": []},
                        ],
                    }
                ]
            },
            "test",
        )

    def test_multiple_messages(self):
        validate_messages_format(
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello!"},
                ]
            },
            "test",
        )

    def test_extra_keys_allowed(self):
        validate_messages_format(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "hi",
                        "tool_calls": None,
                        "refusal": None,
                    }
                ]
            },
            "test",
        )

    def test_string_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_messages_format("hello", "test")

    def test_missing_messages_key(self):
        with pytest.raises(ValueError, match="must contain a 'messages' key"):
            validate_messages_format({"other_key": 1}, "test")

    def test_messages_not_a_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            validate_messages_format({"messages": "not-a-list"}, "test")

    def test_item_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_messages_format({"messages": ["string"]}, "test")

    def test_item_missing_role(self):
        with pytest.raises(ValueError, match="missing required key 'role'"):
            validate_messages_format({"messages": [{"content": "hi"}]}, "test")

    def test_item_missing_content(self):
        with pytest.raises(ValueError, match="missing required key 'content'"):
            validate_messages_format({"messages": [{"role": "user"}]}, "test")

    def test_role_not_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_messages_format({"messages": [{"role": 123, "content": "hi"}]}, "test")

    def test_content_invalid_type_rejected(self):
        with pytest.raises(ValueError, match="must be a string, list, or None"):
            validate_messages_format({"messages": [{"role": "user", "content": 123}]}, "test")

    def test_list_items_rejected(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_messages_format({"messages": [["user", "hello"]]}, "test")

    def test_label_included_in_error(self):
        with pytest.raises(ValueError, match="trace input"):
            validate_messages_format("bad", "trace input")


class TestExtractLastUserMessage:
    def test_multiple_messages_returns_last_user(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "What about Germany?"},
            ]
        }
        result = extract_last_user_message(data)
        assert result == {"messages": [{"role": "user", "content": "What about Germany?"}]}

    def test_single_user_message(self):
        data = {"messages": [{"role": "user", "content": "hello"}]}
        result = extract_last_user_message(data)
        assert result == {"messages": [{"role": "user", "content": "hello"}]}

    def test_no_user_message_returns_unchanged(self):
        data = {"messages": [{"role": "assistant", "content": "hi"}]}
        result = extract_last_user_message(data)
        assert result is data

    def test_non_dict_returns_unchanged(self):
        assert extract_last_user_message("hello") == "hello"
        assert extract_last_user_message(42) == 42
        assert extract_last_user_message(None) is None

    def test_no_messages_key_returns_unchanged(self):
        data = {"query": "hello"}
        assert extract_last_user_message(data) is data

    def test_messages_not_list_returns_unchanged(self):
        data = {"messages": "not-a-list"}
        assert extract_last_user_message(data) is data

    def test_human_role_recognized(self):
        data = {
            "messages": [
                {"role": "human", "content": "hello from langchain"},
            ]
        }
        result = extract_last_user_message(data)
        assert result == {"messages": [{"role": "human", "content": "hello from langchain"}]}

    def test_prefers_last_user_over_earlier_human(self):
        data = {
            "messages": [
                {"role": "human", "content": "first"},
                {"role": "user", "content": "second"},
            ]
        }
        result = extract_last_user_message(data)
        assert result == {"messages": [{"role": "user", "content": "second"}]}


class TestExtractLastAssistantMessage:
    def test_multiple_messages_returns_last_assistant(self):
        data = {
            "messages": [
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "second reply"},
            ]
        }
        result = extract_last_assistant_message(data)
        assert result == {"messages": [{"role": "assistant", "content": "second reply"}]}

    def test_single_assistant_message(self):
        data = {"messages": [{"role": "assistant", "content": "hello"}]}
        result = extract_last_assistant_message(data)
        assert result == {"messages": [{"role": "assistant", "content": "hello"}]}

    def test_no_assistant_message_returns_unchanged(self):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        result = extract_last_assistant_message(data)
        assert result is data

    def test_non_dict_returns_unchanged(self):
        assert extract_last_assistant_message("hello") == "hello"
        assert extract_last_assistant_message(42) == 42
        assert extract_last_assistant_message(None) is None

    def test_no_messages_key_returns_unchanged(self):
        data = {"result": "hello"}
        assert extract_last_assistant_message(data) is data

    def test_messages_not_list_returns_unchanged(self):
        data = {"messages": "not-a-list"}
        assert extract_last_assistant_message(data) is data

    def test_ai_role_recognized(self):
        data = {"messages": [{"role": "ai", "content": "hello from langchain"}]}
        result = extract_last_assistant_message(data)
        assert result == {"messages": [{"role": "ai", "content": "hello from langchain"}]}

    def test_prefers_last_assistant_over_earlier_ai(self):
        data = {
            "messages": [
                {"role": "ai", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        }
        result = extract_last_assistant_message(data)
        assert result == {"messages": [{"role": "assistant", "content": "second"}]}
