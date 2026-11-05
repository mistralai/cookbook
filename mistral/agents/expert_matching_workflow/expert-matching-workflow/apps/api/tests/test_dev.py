from pathlib import Path
from typing import Any

import pytest
import uvicorn
from api import dev


def test_reload_watches_api_and_package_sources_but_no_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: served.update(app=app, **options))

    dev.main()

    watched = [Path(path).relative_to(dev.ROOT).as_posix() for path in served["reload_dirs"]]
    assert served["reload"] is True
    assert "apps/api/src" in watched
    assert "packages/py/env/src" in watched
    assert not any("tests" in Path(path).parts for path in watched)


def test_serves_on_loopback_unless_host_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: served.update(app=app, **options))
    monkeypatch.delenv("HOST", raising=False)

    dev.main()

    assert served["host"] == "127.0.0.1"
