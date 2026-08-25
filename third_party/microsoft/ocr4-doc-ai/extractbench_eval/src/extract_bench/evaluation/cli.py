"""Command-line interface for running evaluation on inference results."""

import json
import sys
from datetime import datetime
from pathlib import Path

import fire

from extract_bench.analysis.detailed_report import generate_detailed_html_report
from extract_bench.evaluation.reports import (
    export_csv as export_csv_report,
)
from extract_bench.evaluation.reports import (
    export_html as export_html_report,
)
from extract_bench.evaluation.reports import (
    export_markdown as export_markdown_report,
)
from extract_bench.evaluation.reports import (
    export_rule_csv as export_rule_csv_report,
)
from extract_bench.evaluation.runner import EvaluationRunner
from extract_bench.schemas.evaluation import EvaluationSummary


class EvaluationCLI:
    """Command-line interface for evaluating inference results."""

    def run(
        self,
        output_dir: str | Path,
        test_cases_dir: str | Path | None = None,
        product_type: str | None = None,
        pipeline_name: str | None = None,
        group: str | None = None,
        report_dir: str | Path | None = None,
        export_csv: bool = True,
        export_rule_csv: bool = True,
        export_markdown: bool = True,
        export_html: bool = True,
        verbose: bool = False,
        force: bool = False,
        multi_task: bool = True,
        max_workers: int | None = None,
        enable_teds: bool = False,
        skip_rules: bool = False,
        ontology: str = "basic",
        verified_only: bool = False,
    ) -> int:
        """
        Run evaluation on inference results.

        Args:
            output_dir: Directory containing inference results
            test_cases_dir: Directory containing test cases (default: inferred from output_dir)
            product_type: Filter by product type (e.g., 'extract', 'parse')
            pipeline_name: Filter by pipeline name (e.g., 'llamaextract_agentic')
            group: Optional group name to filter test cases (e.g., 'arxiv_math')
            report_dir: Directory to save evaluation reports (default: output_dir)
            export_csv: Export results to CSV file (default: False)
            export_markdown: Export summary to markdown file (default: False)
            export_html: Export interactive HTML report (default: False)
            export_rule_csv: Export normalized per-rule results CSV (default: True)
            verbose: Show detailed information about skipped results (default: False)
            force: Force re-evaluation even if results exist (default: False)
            multi_task: Enable multi-task evaluation for mixed rule types (table, order, layout)
            max_workers: Number of parallel workers for evaluation (default: min(CPU count, 8))
            enable_teds: Enable TEDS metric computation in parse evaluation (default: False)
            skip_rules: Skip rule-based metric computation in parse evaluation (default: False)
            ontology: Default ontology for layout evaluation when test case omits ontology
                (e.g. "basic", "canonical")
            verified_only: Discard test_rules explicitly marked verified=false before
                evaluation (default: False)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            output_dir_path = Path(output_dir)
            if not output_dir_path.exists():
                print(f"Error: Output directory does not exist: {output_dir}", file=sys.stderr)
                return 1

            # Infer test_cases_dir and product_type from metadata if not provided
            # First try at output_dir level, then search in subdirectories
            # If a directory has results from multiple product types, it will
            # pick the first one found. This is unlikely in practice since
            # pipelines are generally single-product-type.
            metadata_paths = [output_dir_path / "_metadata.json"]
            # Also check subdirectories (pipeline folders)
            for subdir in output_dir_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("_"):
                    metadata_paths.append(subdir / "_metadata.json")

            for metadata_path in metadata_paths:
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        # Infer test_cases_dir
                        if test_cases_dir is None and "test_cases_dir" in metadata:
                            candidate = Path(metadata["test_cases_dir"])
                            if candidate.exists() and candidate.is_dir():
                                test_cases_dir = candidate
                        # Stop searching once we found valid metadata for test_cases_dir
                        if test_cases_dir is not None:
                            break
                    except Exception:
                        pass  # Ignore errors reading metadata, try next file

            # Infer product_type from actual result files (more reliable than metadata)
            # The metadata may have pipeline's default product_type, but the results
            # may have been produced with auto-detected product_type
            if product_type is None:
                for result_file in output_dir_path.rglob("*.result.json"):
                    try:
                        with open(result_file) as f:
                            result_data = json.load(f)
                        if "product_type" in result_data:
                            product_type = result_data["product_type"]
                            break
                    except Exception:
                        pass

            test_cases_dir_path = Path(test_cases_dir) if test_cases_dir else None
            if verbose and not test_cases_dir_path:
                print(
                    "⚠️  Warning: Could not auto-detect test cases directory. "
                    "Use --test_cases_dir to specify it explicitly."
                )

            # Set report directory
            report_dir_path = Path(report_dir) if report_dir else output_dir_path
            report_dir_path.mkdir(parents=True, exist_ok=True)

            # Create runner
            runner = EvaluationRunner(
                output_dir=output_dir_path,
                test_cases_dir=test_cases_dir_path,
                multi_task=multi_task,
                enable_teds=enable_teds,
                skip_rules=skip_rules,
                layout_ontology=ontology,
                verified_only=verified_only,
            )

            print(f"Running evaluation on: {output_dir_path}")
            if test_cases_dir_path:
                print(f"Test cases directory: {test_cases_dir_path}")
            if product_type:
                print(f"Filtering by product type: {product_type}")
            if pipeline_name:
                print(f"Filtering by pipeline: {pipeline_name}")
            if group:
                print(f"Filtering by group: {group}")
            if verified_only:
                print("Filtering test rules to verified rules only")
            if product_type == "layout_detection" or product_type is None:
                print(f"Default layout ontology: {ontology}")

            # Run evaluation
            summary = runner.run_evaluation(
                product_type=product_type,
                pipeline_name=pipeline_name,
                group=group,
                verbose=verbose,
                max_workers=max_workers,
            )
            summary.completed_at = datetime.now()

            # Save JSON report
            report_json_path = report_dir_path / "_evaluation_report.json"
            report_json_path.write_text(summary.model_dump_json(indent=2))
            print("\n✅ Evaluation complete!")
            print(f"📊 Results saved to: {report_json_path.resolve()}")

            # Print summary
            self._print_summary(summary)

            # Export CSV if requested
            if export_csv:
                csv_path = export_csv_report(summary, report_dir_path)
                print(f"📄 CSV exported to: {csv_path.resolve()}")

            # Export rule-level CSV if requested
            if export_rule_csv:
                rule_csv_path = export_rule_csv_report(
                    summary,
                    report_dir_path,
                    dataset_dir=test_cases_dir_path,
                )
                print(f"🧩 Rule CSV exported to: {rule_csv_path.resolve()}")

            # Export markdown if requested
            if export_markdown:
                md_path = export_markdown_report(summary, report_dir_path)
                print(f"📝 Markdown report exported to: {md_path.resolve()}")

            # Export HTML if requested
            if export_html:
                html_path = export_html_report(summary, report_dir_path)
                print(f"🌐 HTML report exported to: {html_path.resolve()}")

                detailed_html_path = generate_detailed_html_report(
                    summary,
                    report_dir_path,
                    output_dir=output_dir_path,
                    test_cases_dir=test_cases_dir_path,
                    pipeline_name=pipeline_name,
                    group=group,
                )
                print(f"🌐 Detailed HTML report exported to: {detailed_html_path.resolve()}")

            return 0

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    def regenerate_report(
        self,
        evaluation_dir: str | Path,
        test_cases_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        report_dir: str | Path | None = None,
        pdf_base_url: str | None = None,
        export_csv: bool = True,
        export_rule_csv: bool = True,
        export_markdown: bool = True,
        export_html: bool = True,
    ) -> int:
        """Regenerate reports from existing evaluation results without re-running evaluation.

        Useful for re-rendering HTML reports for old runs (e.g. after report format improvements)
        or for regenerating with different options (pdf_base_url, test_cases_dir).

        Args:
            evaluation_dir: Directory containing _evaluation_report.json
                (and usually _metadata.json)
            test_cases_dir: Directory containing test cases (default: inferred from _metadata.json)
            output_dir: Directory containing inference .result.json files (default: evaluation_dir)
            report_dir: Directory to write reports (default: evaluation_dir)
            pdf_base_url: Base URL for PDF files in the HTML report
            export_csv: Export results to CSV file (default: True)
            export_rule_csv: Export normalized per-rule results CSV (default: True)
            export_markdown: Export summary to markdown file (default: True)
            export_html: Export interactive HTML report (default: True)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            evaluation_path = Path(evaluation_dir)
            if not evaluation_path.exists():
                print(
                    f"Error: Evaluation directory does not exist: {evaluation_dir}",
                    file=sys.stderr,
                )
                return 1

            # Load evaluation summary
            summary_json_path = evaluation_path / "_evaluation_report.json"
            if not summary_json_path.exists():
                print(
                    f"Error: {summary_json_path} not found. "
                    "Run 'extract-bench run <pipeline_name>' first to generate results.",
                    file=sys.stderr,
                )
                return 1

            summary_data = json.loads(summary_json_path.read_text())
            summary = EvaluationSummary.model_validate(summary_data)

            # Auto-detect test_cases_dir from _metadata.json if not provided
            metadata_paths = [evaluation_path / "_metadata.json"]
            for subdir in evaluation_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("_"):
                    metadata_paths.append(subdir / "_metadata.json")

            for metadata_path in metadata_paths:
                if metadata_path.exists():
                    try:
                        metadata = json.loads(metadata_path.read_text())
                        if test_cases_dir is None and "test_cases_dir" in metadata:
                            candidate = Path(metadata["test_cases_dir"])
                            if candidate.exists() and candidate.is_dir():
                                test_cases_dir = candidate
                        if test_cases_dir is not None:
                            break
                    except Exception:
                        pass

            test_cases_dir_path = Path(test_cases_dir) if test_cases_dir else None
            output_dir_path = Path(output_dir) if output_dir else evaluation_path
            report_dir_path = Path(report_dir) if report_dir else evaluation_path
            report_dir_path.mkdir(parents=True, exist_ok=True)

            print(f"Regenerating reports from: {summary_json_path.resolve()}")
            print(f"  {summary.total_examples} examples ({summary.successful} successful, {summary.failed} failed)")
            if test_cases_dir_path:
                print(f"  Test cases: {test_cases_dir_path}")
            if pdf_base_url:
                print(f"  PDF base URL: {pdf_base_url}")

            if export_csv:
                csv_path = export_csv_report(summary, report_dir_path)
                print(f"  CSV: {csv_path.resolve()}")

            if export_rule_csv:
                rule_csv_path = export_rule_csv_report(
                    summary,
                    report_dir_path,
                    dataset_dir=test_cases_dir_path,
                )
                print(f"  Rule CSV: {rule_csv_path.resolve()}")

            if export_markdown:
                md_path = export_markdown_report(summary, report_dir_path)
                print(f"  Markdown: {md_path.resolve()}")

            if export_html:
                html_path = export_html_report(summary, report_dir_path)
                print(f"  HTML: {html_path.resolve()}")

                detailed_html_path = generate_detailed_html_report(
                    summary,
                    report_dir_path,
                    output_dir=output_dir_path,
                    test_cases_dir=test_cases_dir_path,
                    pdf_base_url=pdf_base_url,
                )
                print(f"  Detailed HTML: {detailed_html_path.resolve()}")

            print("\nDone!")
            return 0

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    def _print_summary(self, summary: EvaluationSummary) -> None:
        """Print evaluation summary to console."""
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Total Examples:  {summary.total_examples}")
        print(f"Successful:     {summary.successful}")
        print(f"Failed:          {summary.failed}")
        print(f"Skipped:         {summary.skipped}")

        # A run where nothing scored is almost always a broken setup (missing
        # API key, wrong --test_cases_dir, every request erroring) rather than a
        # genuine 0.0. Say so loudly: a quiet empty report reads like success
        # and has already been mistaken for one.
        if summary.successful == 0:
            print("\n" + "!" * 60)
            if summary.total_examples == 0:
                print("WARNING: no documents were scored at all.")
                print("  Nothing matched between the inference output and the test cases.")
                print("  Check that --test_cases_dir points at the downloaded dataset and")
                print("  that the pipeline actually wrote *.result.json files.")
            else:
                print(f"WARNING: all {summary.total_examples} documents scored 0.0.")
                print("  Every document failed inference or produced no usable output,")
                print("  so this run is scored as a total failure rather than skipped.")
                print("  Check the pipeline's _errors.json and your API credentials.")
            print("!" * 60)

        if summary.aggregate_metrics:
            print("\nAggregate Metrics:")
            # Suppress per-doc table count metrics from the summary -- they
            # are surfaced in the detailed report but add clutter here.
            _table_count_avgs = {
                "avg_tables_expected",
                "avg_tables_actual",
                "avg_tables_paired",
                "avg_tables_unmatched_expected",
                "avg_tables_unmatched_pred",
                "avg_tables_unparseable_pred",
            }
            # Print average metrics
            for metric_name, value in sorted(summary.aggregate_metrics.items()):
                if metric_name.startswith("avg_") and metric_name not in _table_count_avgs:
                    print(f"  {metric_name}: {value:.4f}")

            # Print total count metrics
            total_metrics = {
                name: value for name, value in sorted(summary.aggregate_metrics.items()) if name.startswith("total_")
            }
            if total_metrics:
                print("\nTotal Counts:")
                for metric_name, value in sorted(total_metrics.items()):
                    # Format as integer if it's a whole number
                    if value == int(value):
                        print(f"  {metric_name}: {int(value)}")
                    else:
                        print(f"  {metric_name}: {value:.0f}")

        if summary.failed > 0:
            print(f"\n⚠️  {summary.failed} evaluation(s) failed")
            # Show first few errors
            failed_results = [r for r in summary.per_example_results if not r.success]
            for i, result in enumerate(failed_results[:3], 1):
                print(f"\n  {i}. {result.test_id}: {result.error}")

        print("=" * 60)


def main() -> int:
    """Main entry point."""
    cli = EvaluationCLI()
    result = fire.Fire(cli)
    # Fire returns the result of the called method
    # If it's an integer (exit code), use it; otherwise default to 0
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
