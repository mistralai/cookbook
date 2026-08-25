from __future__ import annotations

import json

from extract_bench.evaluation.metrics.extract.array_record_match_metric import (
    _UNHASHABLE,
    _cell_key,
    _intern_field,
    _mismatch_cost_matrix,
)


def _jsonable_intern_key(key: object) -> object:
    """Tuples -> lists so intern keys round-trip through ``json.dumps``."""
    if isinstance(key, tuple):
        return [_jsonable_intern_key(part) for part in key]
    return key


def test_cell_key_interns_json_containers_with_exact_nested_equality() -> None:
    """Opaque JSON arrays and objects intern; nested strings keep exact ``==``."""
    assert _cell_key(["Moscow", "Kyiv"]) is not _UNHASHABLE
    assert _cell_key(["Moscow", "Kyiv"]) == _cell_key(["Moscow", "Kyiv"])
    assert _cell_key(["Moscow", "Kyiv"]) != _cell_key(["Kyiv", "Moscow"])
    assert _cell_key(["Moscow "]) != _cell_key(["Moscow"])
    assert _cell_key("Moscow") != _cell_key(["Moscow"])
    assert _cell_key(["Moscow"]) != _cell_key(("Moscow",))

    assert _cell_key({"city": "Moscow"}) is not _UNHASHABLE
    assert _cell_key({"b": 1, "a": 2}) == _cell_key({"a": 2, "b": 1})
    assert _cell_key({"city": "Moscow "}) != _cell_key({"city": "Moscow"})
    assert _cell_key({}) != _cell_key([])
    assert _cell_key([{"city": "Moscow"}]) == _cell_key([{"city": "Moscow"}])
    assert _cell_key([{"city": "Moscow"}]) != _cell_key([{"city": "Kyiv"}])

    dumped = json.dumps(_jsonable_intern_key(_cell_key({"b": 1, "a": ["x", "y"]})))
    assert json.loads(dumped) == ["d", [[["v", "a"], ["l", [["v", "x"], ["v", "y"]]]], [["v", "b"], ["v", 1]]]]


def test_intern_field_list_column_matches_pairwise_mismatch_cost() -> None:
    actual = [{"addr": ["a", "b"]}, {"addr": ["c"]}]
    expected = [{"addr": ["c"]}, {"addr": ["a", "b"]}]
    interned = _intern_field(actual, expected, "addr")
    assert interned is not None
    cost = _mismatch_cost_matrix(actual, expected, ["addr"], {})
    assert cost[0, 1] == 0
    assert cost[1, 0] == 0
    assert cost[0, 0] == 1
    assert cost[1, 1] == 1


def test_intern_field_dict_column_matches_pairwise_mismatch_cost() -> None:
    actual = [{"amount": {"amount": 4.5, "currency": "USD"}}, {"amount": {"amount": 1.0, "currency": "EUR"}}]
    expected = [{"amount": {"currency": "EUR", "amount": 1.0}}, {"amount": {"currency": "USD", "amount": 4.5}}]
    interned = _intern_field(actual, expected, "amount")
    assert interned is not None
    cost = _mismatch_cost_matrix(actual, expected, ["amount"], {})
    assert cost[0, 1] == 0
    assert cost[1, 0] == 0
    assert cost[0, 0] == 1
    assert cost[1, 1] == 1
