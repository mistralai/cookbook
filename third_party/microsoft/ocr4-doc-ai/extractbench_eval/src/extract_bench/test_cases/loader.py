"""Test case loader for scanning directory structure and loading test cases.

Supports two data formats:
1. JSONL format: {category}.jsonl files with one test case per line (used by extract-bench public dataset)
2. Sidecar format: {pdf_name}.test.json files alongside PDFs
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from extract_bench.test_cases.extract_field_paths import (
    validate_rules_match_expected_output,
)
from extract_bench.test_cases.parse_rule_schemas import coerce_parse_rule_list
from extract_bench.test_cases.schema import (
    ExtractTestCase,
    LayoutDetectionTestCase,
    ParseTestCase,
    QAConfig,
    TestCase,
)

logger = logging.getLogger(__name__)


def _row_identity_kwargs(payload: dict[str, Any], source: Any) -> dict[str, Any]:
    """Extract the eval-only ``_eval_row_identity`` block, refusing a malformed one.

    A typo'd or wrongly-typed block must not be silently ignored: that would
    quietly drop row identity and downgrade the document to gated/positional
    alignment, which is exactly the failure this key exists to prevent.
    """
    if "_eval_row_identity" not in payload:
        legacy = payload.get("_repeated_structure")
        if isinstance(legacy, dict) and legacy:
            logger.warning(
                "%s: ignoring a non-empty _repeated_structure block. Row identity is only read "
                "from _eval_row_identity; honoring the legacy name would activate the inert "
                "blocks that predate it and move scores. Re-run `extract-bench download` to "
                "refresh this sidecar, or rename the key if the declaration is intended.",
                source,
            )
        return {}
    block = payload["_eval_row_identity"]
    if not isinstance(block, dict):
        raise ValueError(f"{source}: _eval_row_identity must be an object, got {type(block).__name__}")
    for array_path, spec in block.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"{source}: _eval_row_identity[{array_path!r}] must be an object, got {type(spec).__name__}"
            )
    return {"_eval_row_identity": block}


# Supported file extensions for input files
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".jfif", ".docx"}

_EXTRACT_RULE_TYPES = frozenset({"extract_field", "array_length", "array_head", "array_tail"})

# Group names become 'schema_field_accuracy_<name>' metrics, so they must be
# safe metric-name fragments.
_METRIC_GROUP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_schema_field_metric_groups(metric_groups: Any, test_json_path: Path) -> None:
    """Validate the v0.2 ``_schema_field_metric_groups`` block (group name -> field paths)."""
    if not isinstance(metric_groups, dict):
        raise ValueError(
            f"Invalid _schema_field_metric_groups in {test_json_path}: must be a dict of group name -> field paths"
        )
    for group_name, paths in metric_groups.items():
        if not isinstance(group_name, str) or not _METRIC_GROUP_NAME_RE.match(group_name):
            raise ValueError(
                f"Invalid _schema_field_metric_groups group name {group_name!r} in {test_json_path}: "
                "must match [a-z][a-z0-9_]* (it becomes the 'schema_field_accuracy_<name>' metric)"
            )
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
            raise ValueError(
                f"Invalid _schema_field_metric_groups[{group_name!r}] in {test_json_path}: "
                "must be a non-empty list of field paths"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{i}, got {type(obj)}")
            rows.append(obj)
    return rows


def _load_jsonl_dataset(root_dir: Path) -> list[TestCase]:
    """Load test cases from extract-bench JSONL format.

    Groups JSONL rows by PDF so inference runs once per unique PDF, not once
    per rule. Each PDF becomes one TestCase with all its rules collected.

    Expected layout:
      <root>/
        {category}.jsonl       # e.g., table.jsonl, chart.jsonl, text.jsonl, layout.jsonl
        expected_markdown.json  # optional: {pdf_path: markdown_string}
        pdfs/{category}/*.pdf

    Each JSONL line has: pdf, page, category, id, type, verified, rule (JSON string or dict)
    """
    jsonl_files = sorted(root_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {root_dir}")

    # Load expected_markdown lookup if available
    expected_markdown_path = root_dir / "expected_markdown.json"
    expected_markdown_map: dict[str, str] = {}
    if expected_markdown_path.exists():
        expected_markdown_map = json.loads(expected_markdown_path.read_text(encoding="utf-8"))

    # First pass: collect all rules grouped by (category, pdf_rel)
    # Key: (category, pdf_rel) -> {category, rules[], layout_rules[], pdf_path, tags}
    pdf_groups: dict[tuple[str, str], dict[str, Any]] = {}

    for jsonl_file in jsonl_files:
        rows = _read_jsonl(jsonl_file)
        for row in rows:
            pdf_rel = row.get("pdf", "")
            category = row.get("category", jsonl_file.stem)
            test_type = row.get("type", "")
            rule_data = row.get("rule", {})

            # Rule may be double-encoded as a JSON string
            if isinstance(rule_data, str):
                rule_data = json.loads(rule_data)

            # Build the rule dict
            rule_dict: dict[str, Any] = {"type": test_type, **rule_data}

            # Preserve original rule id if present
            raw_id = row.get("id")
            if raw_id:
                rule_dict["id"] = raw_id

            # Add page if specified
            page = row.get("page")
            if page is not None:
                rule_dict["page"] = page

            group_key = (category, pdf_rel)
            if group_key not in pdf_groups:
                pdf_groups[group_key] = {
                    "category": category,
                    "parse_rules": [],
                    "layout_rules": [],
                    "expected_markdown": None,
                    "tags": [],
                    "rule_meta": {},
                }

            # Pick up expected_markdown from JSONL row if present
            row_md = row.get("expected_markdown")
            if row_md and pdf_groups[group_key]["expected_markdown"] is None:
                pdf_groups[group_key]["expected_markdown"] = row_md

            # Collect tags from JSONL row
            row_tags = row.get("tags")
            if row_tags and isinstance(row_tags, list):
                for t in row_tags:
                    if t not in pdf_groups[group_key]["tags"]:
                        pdf_groups[group_key]["tags"].append(t)

            # Extract table-specific rule metadata (e.g. allow_splitting_ambiguous_merged_tables)
            for meta_key in ("allow_splitting_ambiguous_merged_tables", "trm_unsupported", "max_top_title_rows"):
                if meta_key in rule_data:
                    pdf_groups[group_key]["rule_meta"][meta_key] = rule_data[meta_key]

            if test_type == "expected_markdown":
                pass  # pointer-only entry, no rule to add
            elif test_type == "layout":
                pdf_groups[group_key]["layout_rules"].append(rule_dict)
            else:
                pdf_groups[group_key]["parse_rules"].append(rule_dict)

    # Categories that share inference results (same PDF, different eval rules).
    # Maps evaluation category -> shared inference directory name.
    _SHARED_INFERENCE_GROUPS = {
        "text_content": "text",
        "text_formatting": "text",
    }

    # Second pass: create one TestCase per unique (category, PDF)
    test_cases: list[TestCase] = []

    for (_, pdf_rel), group_data in pdf_groups.items():
        category = group_data["category"]
        pdf_path = (root_dir / pdf_rel).resolve()
        pdf_stem = Path(pdf_rel).stem
        # Use shared inference group for test_id so results are stored once
        inference_group = _SHARED_INFERENCE_GROUPS.get(category, category)
        test_id = f"{inference_group}/{pdf_stem}"
        # JSONL-inline expected_markdown takes precedence, fall back to JSON file
        expected_md = group_data.get("expected_markdown") or expected_markdown_map.get(pdf_rel)

        parse_rules = group_data["parse_rules"]
        layout_rules = group_data["layout_rules"]
        extra_tags = group_data.get("tags", [])
        rule_meta = group_data.get("rule_meta", {})
        all_tags = [category] + [t for t in extra_tags if t != category]

        if layout_rules and not parse_rules:
            # Pure layout test case
            tc = LayoutDetectionTestCase(
                test_id=test_id,
                group=category,
                file_path=pdf_path,
                tags=all_tags,
                test_rules=layout_rules,
                ontology=layout_rules[0].get("ontology") if layout_rules else None,
                page_index=layout_rules[0].get("page_index", 0) if layout_rules else 0,
            )
        else:
            # Parse test case — only coerce parse rules (layout rules handled separately)
            typed_rules = coerce_parse_rule_list(parse_rules)
            tc = ParseTestCase(
                test_id=test_id,
                group=category,
                file_path=pdf_path,
                tags=all_tags,
                test_rules=typed_rules,
                expected_markdown=expected_md,
                allow_splitting_ambiguous_merged_tables=rule_meta.get("allow_splitting_ambiguous_merged_tables", False),
                trm_unsupported=rule_meta.get("trm_unsupported", False),
                max_top_title_rows=rule_meta.get("max_top_title_rows", 1),
            )
        test_cases.append(tc)

    test_cases.sort(key=lambda tc: tc.test_id)
    return test_cases


def load_test_case(
    file_path: Path,
    test_json_path: Path | None = None,
    *,
    product_type_hint: str | None = None,
) -> TestCase | None:
    """
    Load a single test case for a specific file.

    :param file_path: Path to the input file (PDF, image, etc.)
    :param test_json_path: Optional path to test.json file. If None, looks for
                          `<file_stem>.test.json` in the same directory.
    :return: TestCase if found, None otherwise
    :raises ValueError: If test.json is invalid or missing required fields
    """
    file_path = Path(file_path).resolve()

    # Determine test.json path
    if test_json_path is None:
        test_json_path = file_path.parent / f"{file_path.stem}.test.json"
    else:
        test_json_path = Path(test_json_path).resolve()

    if not test_json_path.exists():
        return None

    # Load test.json
    try:
        with test_json_path.open(encoding="utf-8") as f:
            test_config: dict[str, Any] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in test file {test_json_path}: {e}") from e

    # Generate test_id: group/filename
    group = file_path.parent.name
    test_id = f"{group}/{file_path.stem}"

    # v0.2 _field_rules dict shape (one entry per leaf path) → list shape used by
    # the rest of the loader. Conflict policy: error if both _field_rules and
    # test_rules are present so the test author chooses one shape per file.
    field_rules_dict = test_config.get("_field_rules")
    raw_test_rules = test_config.get("test_rules")
    if field_rules_dict is not None and raw_test_rules is not None:
        raise ValueError(f"{test_json_path}: cannot specify both _field_rules and test_rules; choose one shape")
    test_rules: list[Any]
    if field_rules_dict is not None:
        if not isinstance(field_rules_dict, dict):
            raise ValueError(f"Invalid _field_rules in {test_json_path}: must be a dict keyed by field_path")
        converted: list[Any] = []
        for field_path_key, rule_body in field_rules_dict.items():
            if not isinstance(rule_body, dict):
                raise ValueError(
                    f"Invalid _field_rules entry for {field_path_key!r} in {test_json_path}: must be a dict"
                )
            entry = dict(rule_body)
            existing_type = entry.get("type")
            if existing_type not in (None, "extract_field"):
                raise ValueError(
                    f"_field_rules[{field_path_key!r}].type must be 'extract_field' "
                    f"(got {existing_type!r}) in {test_json_path}"
                )
            entry["type"] = "extract_field"
            entry["field_path"] = field_path_key
            converted.append(entry)
        test_rules = converted
    else:
        test_rules = list(raw_test_rules) if raw_test_rules is not None else []
    if not isinstance(test_rules, list):
        raise ValueError(f"Invalid test_rules field in {test_json_path}: must be a list")
    data_schema = test_config.get("data_schema")

    # Merge tags bidirectionally for compatibility:
    # - top-level test.json tags act as document-scoped tags
    # - those tags propagate into rules so existing rule-level filtering keeps working
    # - explicit per-rule tags bubble back up into the test case level
    # This keeps legacy datasets and reporting behavior stable while still allowing
    # top-level tags to carry document provenance.
    doc_tags = test_config.get("tags", [])
    if not isinstance(doc_tags, list):
        doc_tags = []
    for rule in test_rules:
        rule["tags"] = list(dict.fromkeys(doc_tags + rule.get("tags", [])))
    all_tags = list(dict.fromkeys(doc_tags + [t for rule in test_rules for t in rule.get("tags", [])]))

    # Auto-detect CSV file for chart data array rules
    csv_path = file_path.parent / f"{file_path.stem}.csv"
    for rule in test_rules:
        if rule.get("type") in ("chart_data_array_labels", "chart_data_array_data"):
            rule["_csv_path"] = str(csv_path)

    # Check for layout rules in test_rules
    has_layout_rules = any(r.get("type") == "layout" for r in test_rules)

    has_parse_family_rules = any(
        isinstance(rule, dict) and rule.get("type") not in _EXTRACT_RULE_TYPES for rule in test_rules
    )
    md_sidecar = file_path.parent / f"{file_path.stem}.md"
    has_expected_markdown = test_config.get("expected_markdown") is not None or md_sidecar.exists()
    prefer_parse_annotations = bool(
        product_type_hint
        and product_type_hint.upper() in {"PARSE", "LAYOUT_DETECTION"}
        and (has_parse_family_rules or has_expected_markdown)
    )

    # A schema-bearing sidecar remains an Extract case by default. When a Parse
    # pipeline explicitly loads a dual-use corpus, Parse-native rules or
    # expected markdown take precedence while Extract annotations stay on the
    # mixed ParseTestCase for grounding and provenance.
    if data_schema and not prefer_parse_annotations:
        expected_output = test_config.get("expected_output")
        metric_groups = test_config.get("_schema_field_metric_groups")
        if metric_groups is not None:
            _validate_schema_field_metric_groups(metric_groups, test_json_path)
        extract_case = ExtractTestCase(
            test_id=test_id,
            group=group,
            file_path=file_path,
            tags=all_tags,
            schema=data_schema,
            config=test_config.get("config"),
            expected_output=expected_output,
            test_rules=test_rules,
            **({"_schema_version": test_config["_schema_version"]} if "_schema_version" in test_config else {}),
            **({"_schema_field_metric_groups": metric_groups} if metric_groups is not None else {}),
            **_row_identity_kwargs(test_config, test_json_path),
        )

        # Drift check applies only to legacy rules (no v0.2 evidence list):
        # v0.2 evidence values legitimately differ from expected_output leaves.
        extract_field_rules = [rule for rule in extract_case.get_extract_field_rules() if rule.evidence is None]
        if extract_field_rules and expected_output is not None:
            drifts = validate_rules_match_expected_output(extract_field_rules, expected_output)
            if drifts:
                logger.warning(
                    "extract_field rules disagree with expected_output in %s "
                    "(%d drifts, first 3: %s). Rules are authoritative; "
                    "expected_output is treated as a hint.",
                    test_json_path,
                    len(drifts),
                    drifts[:3],
                )

        return extract_case

    # If layout rules present, create LayoutDetectionTestCase — unless the
    # caller is explicitly loading this corpus for PARSE, in which case the
    # mixed rule list (layout + extract_field + parse rules) loads as a
    # ParseTestCase so parse pipelines can be cross-evaluated on extract
    # grounding rules.
    prefer_parse_case = bool(product_type_hint and product_type_hint.upper() == "PARSE")
    if has_layout_rules and not prefer_parse_case:
        return LayoutDetectionTestCase(
            test_id=test_id,
            group=group,
            file_path=file_path,
            tags=all_tags,
            test_rules=test_rules,
            ontology=test_config.get("ontology"),
            source_ontology=test_config.get("source_ontology"),
            source_dataset=test_config.get("source_dataset"),
            source_id=test_config.get("source_id"),
            page_index=test_config.get("page_index", 0),
            metadata=test_config.get("metadata"),
        )

    # It's a PARSE test case
    # test_rules and expected_markdown are both optional
    expected_markdown = test_config.get("expected_markdown")

    # Auto-load expected_markdown from sidecar .md file if not inline in test.json
    if expected_markdown is None:
        md_sidecar = file_path.parent / f"{file_path.stem}.md"
        if md_sidecar.exists():
            expected_markdown = md_sidecar.read_text(encoding="utf-8")

    # Typed coercion happens in ParseTestCase's validator after all dict
    # mutations (tags, csv_path); it dispatches layout / extract_field /
    # parse rules by discriminator.

    # Check for multiple QA configs (qa_configs array) — stored on the model,
    # expanded into per-question evaluation tasks by the evaluation runner.
    qa_configs = None
    if "qa_configs" in test_config:
        qa_configs_raw = test_config["qa_configs"]
        if not isinstance(qa_configs_raw, list):
            raise ValueError(f"Invalid qa_configs field in {test_json_path}: must be a list")
        qa_configs = [QAConfig.model_validate(qc_raw) for qc_raw in qa_configs_raw]

    # Check for single QA configuration (question-answering test case)
    qa_config = None
    if "qa_config" in test_config:
        # Load QAConfig from nested structure
        qa_config = QAConfig.model_validate(test_config["qa_config"])

    return ParseTestCase(
        test_id=test_id,
        group=group,
        file_path=file_path,
        tags=all_tags,
        test_rules=test_rules if test_rules else None,
        expected_markdown=expected_markdown,
        qa_config=qa_config,
        qa_configs=qa_configs,
        allow_splitting_ambiguous_merged_tables=test_config.get("allow_splitting_ambiguous_merged_tables", False),
        trm_unsupported=test_config.get("trm_unsupported", False),
        max_top_title_rows=test_config.get("max_top_title_rows", 1),
        data_schema=data_schema,
    )


def load_test_cases(
    root_dir: Path,
    require_test_json: bool = False,
    product_type: str | None = None,
) -> list[TestCase]:
    """
    Load all test cases from a directory structure.

    Supports two formats:
    1. JSONL format: Auto-detected if *.jsonl files exist in root_dir
    2. Sidecar format: {pdf_name}.test.json files alongside PDFs

    Directory structure (sidecar):
    <root>/
      <group>/
        <pdf_name>.pdf
        <pdf_name>.test.json

    Directory structure (JSONL):
    <root>/
      {category}.jsonl
      expected_markdown.json
      pdfs/{category}/*.pdf

    :param root_dir: Root directory containing test case groups
    :param require_test_json: If True, skip files without test.json. If False,
                              only load files that have test.json.
    :param product_type: Optional product type filter.

    :return: List of loaded test cases
    :raises ValueError: If root_dir doesn't exist or is invalid
    """
    root_dir = Path(root_dir).resolve()

    if not root_dir.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise ValueError(f"Path is not a directory: {root_dir}")

    # Auto-detect extract-bench JSONL format. Some sidecar datasets include
    # companion *.jsonl artifacts next to PDFs; if sidecar test files exist,
    # prefer the sidecar loader.
    jsonl_files = list(root_dir.glob("*.jsonl"))
    has_sidecar_tests = any(root_dir.rglob("*.test.json"))
    is_extract = product_type and product_type.upper() == "EXTRACT"
    if jsonl_files and not has_sidecar_tests and not is_extract:
        return _load_jsonl_dataset(root_dir)

    test_cases: list[TestCase] = []

    # Determine if test.json is required
    is_layout_detection = product_type and product_type.upper() == "LAYOUT_DETECTION"
    must_have_test_json = require_test_json or is_layout_detection or is_extract

    # Special-case: Standalone test.json files (no PDFs in same dir)
    # This supports multi-task evaluation where tests and PDFs are in different directories
    standalone_test_files = list(root_dir.glob("*.test.json"))
    if standalone_test_files:
        has_source_files = any(
            f.suffix.lower() in SUPPORTED_EXTENSIONS
            for f in root_dir.iterdir()
            if f.is_file() and not f.name.endswith(".test.json")
        )
        if not has_source_files:
            # Load standalone test.json files (works for PARSE and LAYOUT_DETECTION)
            for test_json_path in standalone_test_files:
                doc_name = test_json_path.stem.replace(".test", "")

                # Try to find actual PDF in common locations
                possible_pdf_paths = [
                    root_dir.parent / "pdfs" / f"{doc_name}.pdf",
                    root_dir.parent.parent / "pdfs" / f"{doc_name}.pdf",
                ]
                file_path = None
                for pdf_path in possible_pdf_paths:
                    if pdf_path.exists():
                        file_path = pdf_path
                        break
                if file_path is None:
                    # Use a placeholder path
                    file_path = root_dir / f"{doc_name}.pdf"

                # Use unified load_test_case function
                result = load_test_case(file_path, test_json_path, product_type_hint=product_type)
                if result is not None:
                    if is_extract and not isinstance(result, ExtractTestCase):
                        raise ValueError(f"EXTRACT requires data_schema in {test_json_path}")
                    test_cases.append(result)

            test_cases.sort(key=lambda tc: tc.test_id)
            return test_cases

    # Check if directory has group subdirectories or is flat
    has_group_dirs = any(item.is_dir() for item in root_dir.iterdir())

    # If we have group subdirectories, use structured loading
    if has_group_dirs:
        # Scan for supported files in group subdirectories
        for group_dir in root_dir.iterdir():
            if not group_dir.is_dir():
                continue

            group_name = group_dir.name

            # Find all supported files in this group
            for file_path in group_dir.iterdir():
                if not file_path.is_file():
                    continue

                # Check if file extension is supported
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                # Skip test.json files themselves
                if file_path.name.endswith(".test.json"):
                    continue

                # Try to load test case
                try:
                    result = load_test_case(file_path, product_type_hint=product_type)
                    if result is None:
                        if must_have_test_json:
                            # For LAYOUT_DETECTION, missing test.json is an error
                            if is_extract:
                                raise ValueError(
                                    f"Missing test.json for {file_path}. EXTRACT requires schema from test.json"
                                )
                            if is_layout_detection:
                                raise ValueError(
                                    f"Missing test.json for {file_path}. "
                                    f"LAYOUT_DETECTION requires test_rules with layout annotations"
                                )
                            # For other cases with require_test_json=True, skip
                            continue
                        else:
                            # For PARSE, missing test.json is OK (inference-only)
                            # Create a ParseTestCase without rules
                            group_name = file_path.parent.name
                            test_id = f"{group_name}/{file_path.stem}"
                            result = ParseTestCase(
                                test_id=test_id,
                                group=group_name,
                                file_path=file_path,
                                test_rules=None,
                                expected_markdown=None,
                            )

                    if result is not None:
                        if is_extract and not isinstance(result, ExtractTestCase):
                            raise ValueError(
                                f"EXTRACT requires data_schema in {file_path.parent / f'{file_path.stem}.test.json'}"
                            )
                        test_cases.append(result)
                except ValueError as e:
                    # Invalid test.json - raise error
                    raise ValueError(f"Error loading test case for {file_path}: {e}") from e
                except OSError as e:
                    # File name too long or other OS-level errors - skip with warning
                    print(f"  WARNING: Skipping {file_path.name}: {e}")
                    continue
    else:
        # Flat directory structure - treat root as single group
        # Use root directory name as group, or "root" if it's a path
        group_name = root_dir.name if root_dir.name else "root"

        # Find all supported files in root directory
        for file_path in root_dir.iterdir():
            if not file_path.is_file():
                continue

            # Check if file extension is supported
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Skip test.json files themselves
            if file_path.name.endswith(".test.json"):
                continue

            # Try to load test case
            try:
                result = load_test_case(file_path, product_type_hint=product_type)
                if result is None:
                    if must_have_test_json:
                        # For LAYOUT_DETECTION, missing test.json is an error
                        if is_extract:
                            raise ValueError(
                                f"Missing test.json for {file_path}. EXTRACT requires schema from test.json"
                            )
                        if is_layout_detection:
                            raise ValueError(
                                f"Missing test.json for {file_path}. "
                                f"LAYOUT_DETECTION requires test_rules with layout annotations"
                            )
                        # For other cases with require_test_json=True, skip
                        continue
                    else:
                        # For PARSE, missing test.json is OK (inference-only)
                        # Create a ParseTestCase without rules
                        test_id = f"{group_name}/{file_path.stem}"
                        result = ParseTestCase(
                            test_id=test_id,
                            group=group_name,
                            file_path=file_path,
                            test_rules=None,
                            expected_markdown=None,
                        )

                if result is not None:
                    if is_extract and not isinstance(result, ExtractTestCase):
                        raise ValueError(
                            f"EXTRACT requires data_schema in {file_path.parent / f'{file_path.stem}.test.json'}"
                        )
                    test_cases.append(result)
            except ValueError as e:
                # Invalid test.json - raise error
                raise ValueError(f"Error loading test case for {file_path}: {e}") from e
            except OSError as e:
                # File name too long or other OS-level errors - skip with warning
                print(f"  WARNING: Skipping {file_path.name}: {e}")
                continue

    # Sort by test_id for consistent ordering
    test_cases.sort(key=lambda tc: tc.test_id)

    return test_cases
