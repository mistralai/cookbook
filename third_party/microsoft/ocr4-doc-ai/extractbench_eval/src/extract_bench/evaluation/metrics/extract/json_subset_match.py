"""JSON subset matching metric for extract evaluation.

Ports json_subset_match_score from extract-tests with date normalization support.
"""

import re
from typing import Any

import numpy as np
from autoevals.number import NumericDiff
from autoevals.string import EmbeddingSimilarity, Levenshtein
from dateutil import parser as date_parser
from scipy.optimize import linear_sum_assignment

from extract_bench.evaluation.metrics.field_grounding.evidence_comparator import stable_value_key

_COMMA_WS_RE = re.compile(r"\s*,\s*")


def normalize_date_string(date_str: Any) -> Any:
    """Normalise a date string to ISO format (YYYY-MM-DD).

    Returns the input unchanged if it does not look like a date.
    """
    if not isinstance(date_str, str):
        return date_str

    if len(date_str) < 4 or len(date_str) > 50:
        return date_str

    if date_str.isdigit():
        return date_str

    # Long digit runs (10+) are almost certainly IDs, not dates.
    if re.search(r"\d{10,}", date_str):
        return date_str

    # "March 28,1956" and "March 27 ,1956" are the same date as "March 27, 1956".
    # Normalize on a probe copy only. Comma-only: this runs on every string leaf,
    # and a wider rewrite would widen what counts as a date everywhere.
    probe = _COMMA_WS_RE.sub(", ", date_str) if "," in date_str else date_str

    date_patterns = [
        r"\d{4}-\d{1,2}-\d{1,2}",  # YYYY-MM-DD
        r"\d{1,2}/\d{1,2}/\d{4}",  # MM/DD/YYYY
        r"\d{1,2}/\d{1,2}/\d{2}\b",  # MM/DD/YY (v5 GT canonical format)
        r"\d{1,2}-\d{1,2}-\d{4}",  # MM-DD-YYYY
        r"\d{1,2}-\d{1,2}-\d{2}\b",  # MM-DD-YY
        r"[A-Za-z]+ \d{1,2},? \d{4}",  # Month DD, YYYY
        r"[A-Za-z]+\.? [A-Za-z]+\.? \d{1,2},? \d{4}",  # Weekday Month DD YYYY
        r"\d{1,2} [A-Za-z]+ \d{4}",  # DD Month YYYY
    ]
    if not any(re.search(p, probe) for p in date_patterns):
        return date_str

    try:
        parsed = date_parser.parse(probe, fuzzy=False)
        if parsed.year < 1900 or parsed.year > 2100:
            return date_str
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError, date_parser.ParserError, OverflowError):
        return date_str


# Back-compat alias for the NPI field on rendering_provider. The schema
# rename `rendering_provider_identification_number` -> `provider_npi`
# landed after v5 GT was finalised, so v5 ground truth still uses the
# old key. Normalise both sides to the new key before comparison.
_RENDERING_PROVIDER_KEY = "rendering_provider"
_NPI_OLD_KEY = "rendering_provider_identification_number"
_NPI_NEW_KEY = "provider_npi"


def _normalize_rendering_provider(rp: Any) -> Any:
    """Rename the old NPI key to the new one inside a rendering_provider dict."""
    if not isinstance(rp, dict) or _NPI_OLD_KEY not in rp:
        return rp
    return {(_NPI_NEW_KEY if k == _NPI_OLD_KEY else k): v for k, v in rp.items()}


# Back-compat shapes for payment_details entries seen in v0.3-rough-draft GTs.
# Some GTs use `checks: [<dict>]` (plural list) instead of `check: <dict>`.
# Some GTs put `check_amount` / `check_date` / `payment_method` flat on the
# entry instead of nested under `check`. Some also surface payee information
# as flat `provider_id` / `provider_name` / `provider_address` at the entry
# level instead of nested under `payee`.
#
# Separately, `check_number` was lifted out of the nested `check` object to
# the top level of `PaymentDetails` (LlamaCloud extract API does not yet
# support setting nested identity keys). v5 GTs still carry the nested
# shape, so we lift `check.check_number` to entry["check_number"] on both
# sides before comparison.
_PAYMENT_DETAILS_KEY = "payment_details"
_FLAT_CHECK_FIELDS = ("check_amount", "check_date", "payment_method")


def _normalize_payment_details_entry(entry: Any) -> Any:
    """Rewrite legacy payment_details shapes into the current schema shape."""
    if not isinstance(entry, dict):
        return entry
    out = dict(entry)

    # checks: [<dict>] -> check: <dict>. Take the first element if multi.
    if "checks" in out and "check" not in out:
        checks = out["checks"]
        if isinstance(checks, list) and checks and isinstance(checks[0], dict):
            out["check"] = checks[0]
            out.pop("checks", None)

    # Flat check_* fields -> nested check: {...}.
    flat_check = {k: out.pop(k) for k in _FLAT_CHECK_FIELDS if k in out}
    if flat_check:
        existing_check = out.get("check")
        existing: dict[Any, Any] = existing_check if isinstance(existing_check, dict) else {}
        out["check"] = {**flat_check, **existing}

    # Lift check.check_number to top-level check_number. Schema moved
    # check_number out of the nested Check object; gold files still nest it.
    # Top-level value (if already present) wins over the nested fallback.
    nested_check = out.get("check") if isinstance(out.get("check"), dict) else None
    if nested_check is not None and "check_number" in nested_check:
        nested_value = nested_check.pop("check_number")
        if "check_number" not in out:
            out["check_number"] = nested_value

    # Flat provider_* fields -> nested payee: {...}, only if no existing payee.
    if not isinstance(out.get("payee"), dict):
        payee: dict[str, Any] = {}
        provider_id = out.pop("provider_id", None)
        provider_name = out.pop("provider_name", None)
        provider_address = out.pop("provider_address", None)
        if provider_id is not None:
            if isinstance(provider_id, str) and re.fullmatch(r"\d{10}", provider_id):
                payee["payee_npi"] = provider_id
            else:
                payee["payee_tin"] = provider_id
        if provider_name is not None:
            payee["payee_name"] = provider_name
        if provider_address is not None:
            if isinstance(provider_address, dict):
                payee["payee_address"] = provider_address
            else:
                payee["payee_address"] = {"payee_address_one": provider_address}
        if payee:
            out["payee"] = payee

    return out


def normalize_field_aliases(obj: Any) -> Any:
    """Recursively rewrite back-compat field aliases on both expected and actual."""
    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        for k, v in obj.items():
            if k == _RENDERING_PROVIDER_KEY:
                v = _normalize_rendering_provider(v)
            elif k == _PAYMENT_DETAILS_KEY and isinstance(v, list):
                v = [_normalize_payment_details_entry(e) for e in v]
            result[k] = normalize_field_aliases(v)
        return result
    if isinstance(obj, list):
        return [normalize_field_aliases(item) for item in obj]
    return obj


# Field names that identify a record, so list elements pair by identity
# instead of index. Only unambiguous identifiers: generic ``name`` or
# ``date`` would false-pair.
_IDENTITY_FIELD_NAMES: tuple[str, ...] = (
    "claim_number",
    "claim_id",
    "check_number",
    "invoice_number",
    "payment_id",
    "record_id",
)


def _record_identity(record: Any) -> str | None:
    """Return a normalized identity string for a record, or ``None`` if no
    identity field carries a non-empty value.

    Used by the list-pairing logic so a list of records with mismatched
    *position* but matching *identity* is scored against the right
    counterpart, not the one that happens to live at the same index.
    """
    if not isinstance(record, dict):
        return None
    for k in _IDENTITY_FIELD_NAMES:
        v = record.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s.lower()
        else:
            s = str(v).strip()
            if s:
                return s.lower()
    return None


# A value that is entirely one number (optionally with thousands separators,
# a decimal part, or a trailing dot). Full-match only — substring extraction
# would equate "INV-100" with "100".
_FULL_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _is_nullable_numeric_field(schema_node: Any) -> bool:
    """True when a field's JSON Schema shape is a nullable number.

    Recognizes both idiomatic Pydantic/JSON Schema encodings used across
    extract test cases:

    * ``{"anyOf": [{"type": "number"}, {"type": "null"}], "default": null}``
    * ``{"anyOf": [{"type": "null"}, {"type": "number"}], ...}`` (order-agnostic)
    * ``{"type": ["number", "null"]}`` / ``{"type": ["null", "number"]}``

    Only the shape gate — the ``default`` clause is not required. Callers use
    this to treat ``0`` / ``0.0`` and ``None`` as equivalent on such fields,
    because "not present on this row" and "amount = zero" are not distinguished
    semantically in financial extraction ground truth.
    """
    if not isinstance(schema_node, dict):
        return False
    any_of = schema_node.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2:
        types = {branch.get("type") for branch in any_of if isinstance(branch, dict)}
        if types == {"number", "null"}:
            return True
    type_field = schema_node.get("type")
    if isinstance(type_field, list) and set(type_field) == {"number", "null"}:
        return True
    return False


def _normalize_nullable_numeric(value: Any) -> Any:
    """Collapse ``0`` / ``0.0`` to ``None`` on nullable numeric fields.

    Everything else (non-zero numbers, non-numeric types) is returned as-is.
    Booleans are explicitly excluded from the zero collapse so that ``False``
    is never conflated with an absent value.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value == 0:
        return None
    return value


# Budget for the order-invariant assignment's n×m cost matrix, mirroring the
# v0.2 adapter's ``_ALIGNMENT_PASS2_MAX_PAIRS``. Beyond it (e.g. a 3,000-row
# shuffled array is 9M pairs and ~3s of Python-level cost building, recursing
# into nested lists) index pairing applies instead of stalling the evaluator.
_ASSIGNMENT_MAX_PAIRS = 250_000


def _identity_cell_key(value: Any) -> str:
    """Normalized identity-cell key, aligned with the v0.2 evidence
    alignment's ``stable_value_key`` semantics.

    Surface-form drift in an identity cell (trailing punctuation, casing,
    numeric formatting like ``100`` vs ``"100.00"``) must not unpair the
    row — declared-identity strictness is about *wrong* identity, and the
    drifted cell still pays its own per-field score after pairing. Values
    that are entirely one number canonicalize numerically; everything else
    uses ``stable_value_key``'s text normalization.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(float(value), "g")
    if isinstance(value, str):
        text = value.strip()
        if _FULL_NUMBER_RE.fullmatch(text):
            try:
                return format(float(text.replace(",", "")), "g")
            except ValueError:
                pass
    return stable_value_key(value)


def _declared_identity(record: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    """Composite identity tuple for dataset-declared identity keys.

    Absent and null cells normalize to the empty string — providers may omit
    keys whose value is null, and a null identity cell (e.g. ``put_call``)
    is a legitimate value that still discriminates via the other keys.
    """
    return tuple(_identity_cell_key(record.get(key)) for key in keys)


def _count_leaves(value: Any) -> int:
    """Count leaf nodes in a JSON-like value.

    A leaf is any primitive (str, number, bool, None) or an empty dict / list.
    Used to weight a missing field by the structural cost of what it contains
    in `expected`, so that dropping a 15-claim array isn't scored the same
    as dropping a single scalar.

    Conservative behavior:
    - Empty dict or empty list contribute weight 1 (matches the existing
      "empty == empty" success path which also uses weight 1).
    - Any non-dict / non-list value contributes weight 1.
    """
    if isinstance(value, dict):
        if not value:
            return 1
        return sum(_count_leaves(v) for v in value.values())
    if isinstance(value, list):
        if not value:
            return 1
        return sum(_count_leaves(item) for item in value)
    return 1


def _flatten_normalized_leaves(
    value: Any,
    *,
    case_sensitive: bool,
    normalize_dates: bool,
    _path: tuple[Any, ...] = (),
) -> dict[tuple[Any, ...], Any]:
    """Flatten a JSON-like value into {path: normalized_leaf}.

    Used to build the assignment cost matrix for order-invariant list
    pairing. Strings are normalized the same way the scalar scoring path
    normalizes them (lowercase unless case_sensitive, then date
    normalization) so the approximate cost agrees with the real scorer on
    exact matches. Empty dicts / lists are kept as leaves, matching
    `_count_leaves` semantics.
    """
    if isinstance(value, dict) and value:
        flat: dict[tuple[Any, ...], Any] = {}
        for k, v in value.items():
            flat.update(
                _flatten_normalized_leaves(
                    v, case_sensitive=case_sensitive, normalize_dates=normalize_dates, _path=(*_path, k)
                )
            )
        return flat
    if isinstance(value, list) and value:
        flat = {}
        for i, v in enumerate(value):
            flat.update(
                _flatten_normalized_leaves(
                    v, case_sensitive=case_sensitive, normalize_dates=normalize_dates, _path=(*_path, i)
                )
            )
        return flat
    if isinstance(value, str):
        normalized = value if case_sensitive else value.lower()
        if normalize_dates:
            normalized = normalize_date_string(normalized)
        return {_path: normalized}
    return {_path: value}


def _descend_schema_for_key(schema_node: Any, key: str) -> Any:
    """Return the child schema for ``key`` under ``schema_node`` if declared."""
    if not isinstance(schema_node, dict):
        return None
    props = schema_node.get("properties")
    if isinstance(props, dict) and key in props:
        return props[key]
    return None


def _descend_schema_for_items(schema_node: Any) -> Any:
    """Return the ``items`` schema for an array node."""
    if not isinstance(schema_node, dict):
        return None
    items = schema_node.get("items")
    if isinstance(items, dict):
        return items
    return None


def _compute_score_with_weight(
    expected: Any,
    actual: Any,
    weighted: bool,
    case_sensitive: bool,
    cosine_similarity: bool,
    normalize_dates: bool,
    string_scorer: Any,
    number_scorer: Any,
    identity_keys_by_path: dict[tuple[str, ...], list[str]] | None = None,
    path: tuple[str, ...] = (),
    schema_node: Any = None,
) -> tuple[float, int]:
    """
    Recursively compute match score and weight.

    :param expected: Expected JSON structure
    :param actual: Actual JSON structure
    :param weighted: If True, aggregate by leaf node weights; if False, simple average
    :param case_sensitive: Whether string comparison should be case-sensitive
    :param cosine_similarity: Use embedding similarity for strings
    :param normalize_dates: Normalize date strings before comparison
    :param string_scorer: Scorer for string comparison
    :param number_scorer: Scorer for number comparison
    :param identity_keys_by_path: Dataset-declared identity keys per array
        field path (dict keys only, no list indices); list elements at a
        declared path are paired by composite identity instead of by index
    :param path: Dict-key path from the root to the current value
    :return: (score, weight) where weight is the number of leaf nodes in expected

    When `expected` has structure (dict/list) but `actual` is missing or of
    a non-matching type (most commonly: the key was absent from a parent dict
    so `dict.get(k)` returned None), the score is 0 with weight equal to the
    full leaf count of `expected`. This ensures a dropped 15-claim array is
    penalized by 15 × per-claim leaves rather than weight=1.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            # actual is missing the key entirely (or wrong type). Score 0
            # with weight equal to expected's full leaf count.
            return (0.0, _count_leaves(expected))
        if len(expected) == 0 and len(actual) == 0:
            return (1.0, 1)
        if len(expected) == 0:
            return (1.0, 1)

        # Compute scores and weights for each key
        results: list[tuple[float, int]] = []
        for k in expected.keys():
            score, weight = _compute_score_with_weight(
                expected.get(k),
                actual.get(k),
                weighted=weighted,
                case_sensitive=case_sensitive,
                cosine_similarity=cosine_similarity,
                normalize_dates=normalize_dates,
                string_scorer=string_scorer,
                number_scorer=number_scorer,
                identity_keys_by_path=identity_keys_by_path,
                path=path + (k,),
                schema_node=_descend_schema_for_key(schema_node, k),
            )
            results.append((score, weight))

        if not results:
            return (0.0, 1)

        total_weight = sum(w for _, w in results)
        # When weighted=False, treat each field as weight=1
        effective_weights = [w if weighted else 1 for _, w in results]
        total_eff_weight = sum(effective_weights)
        if total_eff_weight == 0:
            return (0.0, max(total_weight, 1))
        weighted_sum = sum(s * ew for (s, _), ew in zip(results, effective_weights, strict=True))
        agg_score = weighted_sum / total_eff_weight

        return (agg_score, max(total_weight, 1))

    elif isinstance(expected, list):
        if not isinstance(actual, list):
            # actual is missing the key entirely (or wrong type). Score 0
            # with weight equal to expected's full leaf count, recursively.
            return (0.0, _count_leaves(expected))
        if len(expected) == 0 and len(actual) == 0:
            return (1.0, 1)
        if len(expected) == 0:
            return (1.0, 1)
        if len(actual) == 0:
            # All expected items missing - weight by expected's full leaf
            # count so a dropped 15-claim array is penalized by 15 × per-claim
            # leaves, not just len(expected).
            return (0.0, max(_count_leaves(expected), 1))

        # Dataset-declared identity pairing: when the test case declares
        # identity keys for this array (the ``match_by:<keys>`` structural
        # rule), pair each expected row with the predicted row carrying the
        # same composite identity. Duplicate identities claim predictions in
        # occurrence order so relative order is preserved among them.
        declared_keys = identity_keys_by_path.get(path) if identity_keys_by_path else None
        item_schema = _descend_schema_for_items(schema_node)
        if declared_keys and all(isinstance(item, dict) for item in expected):
            actual_by_declared: dict[tuple[str, ...], list[Any]] = {}
            for item in actual:
                if isinstance(item, dict):
                    actual_by_declared.setdefault(_declared_identity(item, declared_keys), []).append(item)
            declared_results: list[tuple[float, int]] = []
            for exp_item in expected:
                bucket = actual_by_declared.get(_declared_identity(exp_item, declared_keys))
                if bucket:
                    score, weight = _compute_score_with_weight(
                        exp_item,
                        bucket.pop(0),
                        weighted=weighted,
                        case_sensitive=case_sensitive,
                        cosine_similarity=cosine_similarity,
                        normalize_dates=normalize_dates,
                        string_scorer=string_scorer,
                        number_scorer=number_scorer,
                        identity_keys_by_path=identity_keys_by_path,
                        path=path,
                        schema_node=item_schema,
                    )
                    declared_results.append((score, weight))
                else:
                    declared_results.append((0.0, _count_leaves(exp_item)))

            total_weight = sum(w for _, w in declared_results) or 1
            if weighted:
                agg_score = sum(s * w for s, w in declared_results) / total_weight
            else:
                agg_score = sum(s for s, _ in declared_results) / max(len(expected), len(actual))
            return (agg_score, max(total_weight, 1))

        # Identity-keyed pairing: when every expected element carries an
        # identity field (claim_number, payment_id, etc.) the right
        # comparison counterpart is the actual element with the same identity,
        # not the one with the most overlapping fields. Falls back to
        # order-invariant assignment pairing when expected has no identity
        # or the identities are mixed.
        exp_identities = [_record_identity(item) for item in expected]
        if exp_identities and all(i is not None for i in exp_identities):
            actual_by_identity: dict[str, Any] = {}
            unclaimed_actual: list[Any] = []
            for item in actual:
                aid = _record_identity(item)
                if aid is not None and aid not in actual_by_identity:
                    actual_by_identity[aid] = item
                else:
                    unclaimed_actual.append(item)
            list_results: list[tuple[float, int]] = []
            for i, exp_item in enumerate(expected):
                eid = exp_identities[i]
                if eid in actual_by_identity:
                    score, weight = _compute_score_with_weight(
                        exp_item,
                        actual_by_identity[eid],
                        weighted=weighted,
                        case_sensitive=case_sensitive,
                        cosine_similarity=cosine_similarity,
                        normalize_dates=normalize_dates,
                        string_scorer=string_scorer,
                        number_scorer=number_scorer,
                        identity_keys_by_path=identity_keys_by_path,
                        path=path,
                        schema_node=item_schema,
                    )
                    list_results.append((score, weight))
                else:
                    list_results.append((0.0, _count_leaves(exp_item)))

            if not list_results:
                return (0.0, 1)

            total_weight = sum(w for _, w in list_results) or 1
            if weighted:
                agg_score = sum(s * w for s, w in list_results) / total_weight
            else:
                agg_score = sum(s for s, _ in list_results) / max(len(expected), len(actual))
            return (agg_score, max(total_weight, 1))

        # Order-invariant pairing: optimal one-to-one assignment between
        # expected and actual elements (Hungarian algorithm), mirroring
        # ArrayRecordMatchMetric. The assignment cost is an approximate
        # mismatch count over pre-normalized leaves (cheap exact equality,
        # computed once per element so the n×m matrix stays cheap); the
        # score for each assigned pair is then the full recursive
        # partial-credit score. Unmatched expected elements score 0 with
        # their full leaf weight; extra actual elements are ignored in
        # weighted mode (subset semantics) and penalized through the
        # max-length denominator in unweighted mode.
        if len(expected) * len(actual) > _ASSIGNMENT_MAX_PAIRS:
            # Pathologically long arrays: building the cost matrix is
            # O(n·m·leaves) in Python, so beyond the pair budget pair by
            # index instead of stalling the evaluator.
            assigned = {i: i for i in range(min(len(expected), len(actual)))}
        else:
            expected_flat = [
                _flatten_normalized_leaves(item, case_sensitive=case_sensitive, normalize_dates=normalize_dates)
                for item in expected
            ]
            actual_flat = [
                _flatten_normalized_leaves(item, case_sensitive=case_sensitive, normalize_dates=normalize_dates)
                for item in actual
            ]
            # Tie-break toward index order: scipy does not document which
            # optimal assignment it returns when several are tied (e.g.
            # every row is a near-miss and the cost matrix is uniform), so
            # without this a scipy upgrade could legally swap aligned
            # near-miss rows for crossed ones and silently drop their
            # Levenshtein partial credit. The index distance term is scaled
            # so its total over any assignment stays below 1 (sum of |i-j|
            # is bounded by n*m), which means it can never override a real
            # integer mismatch-count difference.
            tiebreak_eps = 1.0 / (len(expected) * len(actual) + 1)
            cost = np.empty((len(expected), len(actual)))
            for i, exp_leaves in enumerate(expected_flat):
                for j, act_leaves in enumerate(actual_flat):
                    mismatches = sum(
                        1
                        for leaf_path, leaf in exp_leaves.items()
                        if leaf_path not in act_leaves or act_leaves[leaf_path] != leaf
                    )
                    cost[i, j] = mismatches + tiebreak_eps * abs(i - j)
            exp_indices, act_indices = linear_sum_assignment(cost)
            assigned = dict(zip(exp_indices.tolist(), act_indices.tolist(), strict=True))

        list_results = []
        for i, exp_item in enumerate(expected):
            paired_index = assigned.get(i)
            if paired_index is None:
                # More expected than actual: this element went unmatched.
                # Weight by full leaf count so dropping rows of a long
                # array isn't under-penalized.
                list_results.append((0.0, _count_leaves(exp_item)))
                continue
            score, weight = _compute_score_with_weight(
                exp_item,
                actual[paired_index],
                weighted=weighted,
                case_sensitive=case_sensitive,
                cosine_similarity=cosine_similarity,
                normalize_dates=normalize_dates,
                string_scorer=string_scorer,
                number_scorer=number_scorer,
                identity_keys_by_path=identity_keys_by_path,
                path=path,
                schema_node=item_schema,
            )
            list_results.append((score, weight))

        if not list_results:
            return (0.0, 1)

        total_weight = sum(w for _, w in list_results)
        if weighted:
            # Weighted: each element contributes proportionally to its leaf count
            if total_weight == 0:
                return (0.0, 1)
            agg_score = sum(s * w for s, w in list_results) / total_weight
        else:
            # Unweighted: divide by max length to penalize extra items in actual
            agg_score = sum(s for s, _ in list_results) / max(len(expected), len(actual))

        return (agg_score, max(total_weight, 1))

    elif isinstance(expected, str):
        if not isinstance(actual, str):
            return (0.0, 1)

        expected_normalized = expected
        actual_normalized = actual

        if not case_sensitive:
            expected_normalized = expected_normalized.lower()
            actual_normalized = actual_normalized.lower()

        if normalize_dates:
            expected_normalized = normalize_date_string(expected_normalized)
            actual_normalized = normalize_date_string(actual_normalized)

        result = string_scorer.eval(expected_normalized, actual_normalized)
        score = result.score if hasattr(result, "score") else 0.0
        return (score, 1)

    elif isinstance(expected, (int, float)):
        # Nullable numeric fields treat 0 / 0.0 and None as equivalent — the
        # semantic on financial extraction ground truth is that null means
        # "not present on this row" and there is no meaningful distinction
        # between "not present" and "amount = zero". Gated strictly on the
        # JSON Schema shape so plain numeric fields keep strict semantics.
        # Booleans are excluded from the collapse (they are a subclass of int
        # but ``False`` is never conflated with an absent value).
        if not isinstance(expected, bool) and _is_nullable_numeric_field(schema_node):
            norm_expected = _normalize_nullable_numeric(expected)
            norm_actual = _normalize_nullable_numeric(actual)
            if norm_expected is None and norm_actual is None:
                return (1.0, 1)
        if not isinstance(actual, (int, float)):
            return (0.0, 1)
        result = number_scorer.eval(expected, actual)
        score = result.score if hasattr(result, "score") else 0.0
        return (score, 1)

    elif expected is None:
        if actual is None:
            return (1.0, 1)
        # Same nullable-numeric gate for the None-expected direction: 0/0.0
        # predicted against a null-expected cell scores as a match.
        if (
            _is_nullable_numeric_field(schema_node)
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and _normalize_nullable_numeric(actual) is None
        ):
            return (1.0, 1)
        return (0.0, 1)

    else:
        # Type mismatch or unsupported type
        return (0.0, 1)


def json_subset_match_score(
    expected: Any,
    actual: Any,
    case_sensitive: bool = True,
    cosine_similarity: bool = False,
    normalize_dates: bool = True,
    weighted: bool = True,
    identity_keys_by_path: dict[tuple[str, ...], list[str]] | None = None,
    data_schema: dict[str, Any] | None = None,
) -> float:
    """
    Calculate similarity score between expected and actual JSON structures.

    Adapted from autoevals.JsonDiff to only test on the subset of keys within
    the expected json. This means extra keys in actual are ignored.

    :param expected: Expected JSON structure (dict, list, or primitive)
    :param actual: Actual JSON structure to compare
    :param case_sensitive: Whether string comparison should be case-sensitive
    :param cosine_similarity: Use embedding similarity for strings (slower but more semantic)
    :param normalize_dates: Normalize date strings before comparison
    :param weighted: If True (default), weight fields by their number of leaf nodes.
                     If False, use simple averaging (each field/element counts equally).
    :param identity_keys_by_path: Dataset-declared identity keys per array field
        path (dict-key tuples, e.g. ``{("line_items",): ["item_no"]}``); list
        elements at a declared path are paired by composite identity instead of
        by index, so reordered or dropped rows fail only themselves
    :return: Similarity score between 0.0 and 1.0
    """
    string_scorer = Levenshtein() if not cosine_similarity else EmbeddingSimilarity()
    number_scorer = NumericDiff()

    # Normalize back-compat field aliases on both sides before comparison.
    # Handles rendering_provider NPI rename and legacy payment_details shapes
    # (checks-plural, flat check_* fields, nested check_number lifted to top).
    expected = normalize_field_aliases(expected)
    actual = normalize_field_aliases(actual)

    score, _ = _compute_score_with_weight(
        expected=expected,
        actual=actual,
        weighted=weighted,
        case_sensitive=case_sensitive,
        cosine_similarity=cosine_similarity,
        normalize_dates=normalize_dates,
        string_scorer=string_scorer,
        number_scorer=number_scorer,
        identity_keys_by_path=identity_keys_by_path,
        schema_node=data_schema,
    )
    return score
