"""Pipeline-name filtering must match whole path components, not substrings.

Several registry names are prefixes of other registry names
(``llamaextract_agentic`` / ``llamaextract_agentic_plus`` /
``llamaextract_agentic_standard_bbox``, ``extend_extract`` /
``extend_extract_max``). A substring filter silently mixes the longer
pipeline's results and inference failures into the shorter pipeline's score.
"""

import json

from extract_bench.evaluation.runner import EvaluationRunner

COLLIDING = [
    "llamaextract_agentic",
    "llamaextract_agentic_plus",
    "llamaextract_agentic_standard_bbox",
]


def _build_output_tree(root, pipelines):
    """Create ``<root>/<pipeline>/short/doc.result.json`` + ``_errors.json``."""
    for pipe in pipelines:
        split_dir = root / pipe / "short"
        split_dir.mkdir(parents=True)
        (split_dir / f"{pipe}_doc.result.json").write_text("{}", encoding="utf-8")
        (root / pipe / "_metadata.json").write_text("{}", encoding="utf-8")
        (root / pipe / "_errors.json").write_text(
            json.dumps([{"example_id": f"short/{pipe}_failed", "error": "boom"}]),
            encoding="utf-8",
        )


def test_load_inference_errors_ignores_prefix_siblings(tmp_path):
    _build_output_tree(tmp_path, COLLIDING)
    runner = EvaluationRunner.__new__(EvaluationRunner)

    errors = runner._load_inference_errors(tmp_path, pipeline_name="llamaextract_agentic")

    assert [e["example_id"] for e in errors] == ["short/llamaextract_agentic_failed"]


def test_load_inference_errors_unfiltered_returns_all(tmp_path):
    _build_output_tree(tmp_path, COLLIDING)
    runner = EvaluationRunner.__new__(EvaluationRunner)

    errors = runner._load_inference_errors(tmp_path, pipeline_name=None)

    assert len(errors) == len(COLLIDING)
    assert {e["_pipeline_dir"] for e in errors} == set(COLLIDING)


def test_load_inference_errors_when_output_dir_is_the_pipeline_dir(tmp_path):
    """The documented flow passes ``output/<pipeline>`` directly as output_dir."""
    _build_output_tree(tmp_path, COLLIDING)
    runner = EvaluationRunner.__new__(EvaluationRunner)

    errors = runner._load_inference_errors(tmp_path / "llamaextract_agentic", pipeline_name="llamaextract_agentic")

    assert [e["example_id"] for e in errors] == ["short/llamaextract_agentic_failed"]


def test_result_file_filter_ignores_prefix_siblings(tmp_path):
    """The result-file filter is the same component match as the errors filter."""
    _build_output_tree(tmp_path, COLLIDING)
    pipeline_name = "llamaextract_agentic"

    all_results = sorted(tmp_path.rglob("*.result.json"))
    kept = [f for f in all_results if pipeline_name in f.parent.parts]

    assert len(all_results) == len(COLLIDING)
    assert [f.name for f in kept] == ["llamaextract_agentic_doc.result.json"]
