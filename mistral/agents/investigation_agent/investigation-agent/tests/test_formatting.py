import pytest

from formatting import agent_reply_text


@pytest.mark.parametrize(
    ("markdown", "mrkdwn"),
    [
        ("## **Cause**\n\nSee **this** and ~~that~~ <@U1>.\n\n---\n- [docs](https://x.io/a) ![img](https://x.io/i.png) [file](src/a.py)", "*Cause*\n\nSee *this* and ~that~ .\n\n• <https://x.io/a|docs> https://x.io/i.png file"),
        # Code and URLs stay verbatim: Slack shows code literally, and a rewritten URL is a dead link.
        ("The crash is in `Checkout.__init__`, called from `__main__`.", "The crash is in `Checkout.__init__`, called from `__main__`."),
        ("```python\n# Whole dollars\ndef wrap(*args, **kwargs):\n- return int(value)\n```", "```python\n# Whole dollars\ndef wrap(*args, **kwargs):\n- return int(value)\n```"),
        ("`2 ** 10` overflows.\n\n**Evidence**\n- https://github.com/a/b/blob/1/b/__main__.py#L3", "`2 ** 10` overflows.\n\n*Evidence*\n• https://github.com/a/b/blob/1/b/__main__.py#L3"),
        ("See [`pricing/__init__.py`](https://github.com/a/b/blob/1/pricing/__init__.py#L12).", "See <https://github.com/a/b/blob/1/pricing/__init__.py#L12|`pricing/__init__.py`>."),
    ],
)
def test_agent_reply_text(markdown: str, mrkdwn: str) -> None:
    assert agent_reply_text(markdown) == mrkdwn
