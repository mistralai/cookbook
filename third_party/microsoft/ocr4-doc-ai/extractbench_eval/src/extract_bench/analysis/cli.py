"""Command-line interface for analysis tools."""

import json
import sys
from pathlib import Path

import fire

from extract_bench.analysis.aggregation_report import generate_aggregation_report
from extract_bench.analysis.comparison import PipelineComparison
from extract_bench.analysis.comparison_report import (
    generate_comparison_html,
    resolve_comparison_pdf_base_url,
)
from extract_bench.analysis.detailed_report import generate_detailed_html_report
from extract_bench.analysis.leaderboard_report import generate_leaderboard_report
from extract_bench.schemas.evaluation import EvaluationSummary
from extract_bench.utils.browser import open_browser


class AnalysisCLI:
    """Command-line interface for analyzing and comparing pipeline results."""

    def compare_pipelines(
        self,
        pipeline_a_dir: str | Path,
        pipeline_b_dir: str | Path,
        test_cases_dir: str | Path | None = None,
        output_file: str | Path | None = None,
        pdf_base_url: str | None = None,
    ) -> int:
        """
        Compare results from two different pipelines.

        Args:
            pipeline_a_dir: Directory containing pipeline A evaluation results
            pipeline_b_dir: Directory containing pipeline B evaluation results
            test_cases_dir: Optional directory containing test cases (for input files and schemas)
            output_file: Path to save the comparison HTML report
                (default: pipeline_a_dir/comparison.html)
            pdf_base_url: Optional base URL/path for PDF preview in the report.
                When omitted, a relative path from the report to test_cases_dir is computed.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            pipeline_a_path = Path(pipeline_a_dir)
            pipeline_b_path = Path(pipeline_b_dir)

            if not pipeline_a_path.exists():
                print(
                    f"Error: Pipeline A directory does not exist: {pipeline_a_path}",
                    file=sys.stderr,
                )
                return 1

            if not pipeline_b_path.exists():
                print(
                    f"Error: Pipeline B directory does not exist: {pipeline_b_path}",
                    file=sys.stderr,
                )
                return 1

            # Auto-detect test_cases_dir if not provided
            if test_cases_dir is None:
                # Try to get from pipeline A metadata
                metadata_path = pipeline_a_path / "_metadata.json"
                if metadata_path.exists():
                    try:
                        import json

                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        if "test_cases_dir" in metadata:
                            candidate = Path(metadata["test_cases_dir"])
                            if candidate.exists() and candidate.is_dir():
                                test_cases_dir = candidate
                    except Exception:
                        pass

            test_cases_path = Path(test_cases_dir) if test_cases_dir else None

            print("Comparing pipelines:")
            print(f"  Pipeline A: {pipeline_a_path}")
            print(f"  Pipeline B: {pipeline_b_path}")
            if test_cases_path:
                print(f"  Test Cases: {test_cases_path}")

            # Run comparison
            comparison = PipelineComparison(
                pipeline_a_dir=pipeline_a_path,
                pipeline_b_dir=pipeline_b_path,
                test_cases_dir=test_cases_path,
            )

            print("\nLoading and comparing results...")
            comparison_data = comparison.compare()

            stats = comparison_data["stats"]
            print("\nComparison Results:")
            print(f"  Total Matched: {stats['total_matched']}")
            print(f"  {stats['pipeline_a_name']} Better: {stats['a_better']}")
            print(f"  {stats['pipeline_b_name']} Better: {stats['b_better']}")
            print(f"  Both Bad: {stats['both_bad']}")
            print(f"  Tie: {stats['tie']}")

            # Generate HTML report
            if output_file is None:
                output_file = pipeline_a_path / "comparison.html"
            else:
                output_file = Path(output_file)

            comparison_data["pdf_base_url"] = resolve_comparison_pdf_base_url(
                test_cases_path,
                output_file,
                explicit_url=pdf_base_url,
            )

            print("\nGenerating comparison report...")
            report_path = generate_comparison_html(comparison_data, output_file)

            print(f"\n✓ Comparison report saved to: {report_path.absolute()}")  # type: ignore[union-attr]
            print("  Open in browser to view interactive comparison")
            if comparison_data["pdf_base_url"]:
                print(
                    "  PDF preview base: "
                    f"{comparison_data['pdf_base_url']} "
                    "(serve repo root, e.g. extract-bench serve .)"
                )

            return 0
        except Exception as e:
            import traceback

            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1

    def generate_report(
        self,
        evaluation_dir: str | Path,
        test_cases_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        output_file: str | Path | None = None,
        pdf_base_url: str | None = None,
        pipeline_name: str | None = None,
        group: str | None = None,
    ) -> int:
        """
        Generate a detailed interactive HTML report from evaluation results.

        This loads the evaluation summary JSON and generates an interactive HTML report
        with drill-down capabilities for each test case, showing input files, outputs,
        and metrics.

        Args:
            evaluation_dir: Directory containing evaluation results
                (should have _evaluation_report.json)
            test_cases_dir: Optional directory containing test cases
                (for input files and schemas)
            output_dir: Directory containing inference results
                (*.result.json files). If not provided, defaults to
                evaluation_dir. Use this when evaluation results are
                stored separately.
            output_file: Path to save the HTML report
                (default: evaluation_dir/_evaluation_report_detailed.html)
            pdf_base_url: Base URL for PDF files (e.g., http://localhost:8080/data).
                          If provided, this URL is pre-populated in the report for viewing PDFs.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            evaluation_path = Path(evaluation_dir)

            if not evaluation_path.exists():
                print(
                    f"Error: Evaluation directory does not exist: {evaluation_path}",
                    file=sys.stderr,
                )
                return 1

            # Check for _evaluation_report.json at top level (single-category)
            summary_json_path = evaluation_path / "_evaluation_report.json"

            if not summary_json_path.exists():
                # Auto-detect multi-category: look for subdirectories with reports
                category_dirs = sorted(
                    d
                    for d in evaluation_path.iterdir()
                    if d.is_dir() and not d.name.startswith("_") and (d / "_evaluation_report.json").exists()
                )
                if category_dirs:
                    print(
                        f"Multi-category output detected. Generating reports for: "
                        f"{', '.join(d.name for d in category_dirs)}"
                    )
                    generated = []
                    for cat_dir in category_dirs:
                        print(f"\n--- {cat_dir.name} ---")
                        ret = self.generate_report(
                            evaluation_dir=str(cat_dir),
                            test_cases_dir=test_cases_dir,
                            output_dir=str(cat_dir) if output_dir is None else output_dir,
                            output_file=None,
                            pdf_base_url=pdf_base_url,
                        )
                        if ret == 0:
                            generated.append(cat_dir.name)
                    print(f"\n✓ Generated reports for: {', '.join(generated)}")
                    return 0
                else:
                    print(
                        f"Error: Evaluation report not found: {summary_json_path}",
                        file=sys.stderr,
                    )
                    print(
                        "  No per-category reports found either. Run evaluation first.",
                        file=sys.stderr,
                    )
                    return 1

            print(f"Loading evaluation summary from: {summary_json_path}")
            with open(summary_json_path) as f:
                summary_data = json.load(f)
            summary = EvaluationSummary.model_validate(summary_data)

            # Auto-detect test_cases_dir if not provided
            if test_cases_dir is None:
                metadata_path = evaluation_path / "_metadata.json"
                if not metadata_path.exists():
                    # Check parent for multi-category layout
                    metadata_path = evaluation_path.parent / "_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        if "test_cases_dir" in metadata:
                            candidate = Path(metadata["test_cases_dir"])
                            if candidate.exists() and candidate.is_dir():
                                test_cases_dir = candidate
                    except Exception:
                        pass

            test_cases_path = Path(test_cases_dir) if test_cases_dir else None

            # Determine output_dir (where inference *.result.json files are)
            if output_dir is None:
                metadata_path = evaluation_path / "_metadata.json"
                if not metadata_path.exists():
                    metadata_path = evaluation_path.parent / "_metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        if "output_dir" in metadata:
                            candidate = Path(metadata["output_dir"])
                            if candidate.exists() and candidate.is_dir():
                                output_dir = candidate
                    except Exception:
                        pass
                if output_dir is None:
                    output_dir = evaluation_path
            output_path = Path(output_dir)

            # Determine output file
            if output_file is None:
                output_file = evaluation_path / "_evaluation_report_detailed.html"
            else:
                output_file = Path(output_file)

            print("Generating detailed HTML report...")
            print(f"  Evaluation dir: {evaluation_path}")
            print(f"  Output dir (inference): {output_path}")
            if test_cases_path:
                print(f"  Test cases dir: {test_cases_path}")
            print(f"  Output file: {output_file}")

            # Generate report
            report_path = generate_detailed_html_report(
                summary=summary,
                report_dir=evaluation_path,
                output_dir=output_path,
                test_cases_dir=test_cases_path,
                pdf_base_url=pdf_base_url,
                pipeline_name=pipeline_name,
                group=group,
            )

            print(f"\n✓ Detailed report saved to: {report_path.absolute()}")
            print("  Open in browser to view interactive report")

            return 0
        except Exception as e:
            import traceback

            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1

    def generate_leaderboard(
        self,
        output_dir: str | Path = "./output",
        pipelines: list[str] | None = None,
        output_file: str | Path | None = None,
    ) -> int:
        """Generate a leaderboard comparing all pipelines side-by-side.

        Args:
            output_dir: Parent directory containing pipeline subdirectories (default: ./output)
            pipelines: Optional list of pipeline directory names to include.
                If not provided, auto-discovers all pipelines in output_dir.
            output_file: Path to save the leaderboard HTML
                (default: output_dir/_leaderboard.html)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                print(f"Error: Output directory does not exist: {output_path}", file=sys.stderr)
                return 1

            pipeline_names = list(pipelines) if pipelines else None
            out_file = Path(output_file) if output_file else None

            print(f"Scanning for pipelines in: {output_path}")
            report_path = generate_leaderboard_report(
                output_dir=output_path,
                pipeline_names=pipeline_names,
                output_file=out_file,
            )

            print(f"\n✓ Leaderboard saved to: {report_path.absolute()}")
            report_url = report_path.absolute().as_uri()
            print(f"  Open in browser: {report_url}")
            open_browser(report_url)
            return 0
        except Exception as e:
            import traceback

            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1

    def serve(
        self,
        pipeline_dir: str | Path | None = None,
        port: int = 8080,
        root: str | Path = ".",
    ) -> int:
        """Start a local HTTP server to view reports with PDF rendering support.

        Browsers block file:// access to PDFs for security reasons. This serves
        the project root over HTTP so both reports and PDFs are accessible.

        Args:
            pipeline_dir: Pipeline output directory to open in browser
                (e.g., ./output/llamaextract_agentic). If provided, opens the
                dashboard or detailed report automatically.
            port: Port number (default: 8080)
            root: Root directory to serve (default: current directory).
                Must contain both data/ and output/ subdirectories.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        import http.server
        import os
        import socketserver

        serve_path = Path(root).resolve()
        if not serve_path.exists():
            print(f"Error: Directory does not exist: {serve_path}", file=sys.stderr)
            return 1

        os.chdir(serve_path)
        handler = http.server.SimpleHTTPRequestHandler

        # Find an available port, starting from the requested one
        actual_port = port
        httpd = None
        for attempt_port in range(port, port + 100):
            try:
                httpd = socketserver.TCPServer(("", attempt_port), handler)
                actual_port = attempt_port
                break
            except OSError:
                continue

        if httpd is None:
            print(f"Error: Could not find an available port in range {port}-{port + 99}", file=sys.stderr)
            return 1

        url = f"http://localhost:{actual_port}"

        # Determine what to open in browser
        open_url = url
        if pipeline_dir is not None:
            rel_path = Path(pipeline_dir)
            dashboard = rel_path / "_evaluation_report_dashboard.html"
            detailed = rel_path / "_evaluation_report_detailed.html"
            if dashboard.exists():
                open_url = f"{url}/{dashboard}"
            elif detailed.exists():
                open_url = f"{url}/{detailed}"
            else:
                open_url = f"{url}/{rel_path}"

        print(f"Serving from: {serve_path}")
        print(f"URL:          {url}")
        if actual_port != port:
            print(f"  (port {port} was in use, using {actual_port})")
        print(f"\nOpen in browser: {open_url}")
        print("Press Ctrl+C to stop\n")

        open_browser(open_url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
        finally:
            httpd.server_close()
        return 0

    def generate_dashboard(
        self,
        evaluation_dir: str | Path,
        groups: list[str] | None = None,
        pipeline_name: str = "",
    ) -> int:
        """Generate an aggregation dashboard from per-category evaluation results.

        Args:
            evaluation_dir: Directory containing per-category subdirectories,
                each with _evaluation_report.json.
            groups: List of category names. If not provided, auto-discovers
                subdirectories containing _evaluation_report.json.
            pipeline_name: Pipeline name for display in the report header.

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            eval_path = Path(evaluation_dir)
            if not eval_path.exists():
                print(f"Error: Directory does not exist: {eval_path}", file=sys.stderr)
                return 1

            # Auto-discover groups if not provided
            if groups is None:
                groups = sorted(
                    d.name
                    for d in eval_path.iterdir()
                    if d.is_dir() and not d.name.startswith("_") and (d / "_evaluation_report.json").exists()
                )

            if not groups:
                print("Error: No category evaluation reports found", file=sys.stderr)
                return 1

            print(f"Generating dashboard for categories: {', '.join(groups)}")
            report_path = generate_aggregation_report(
                pipeline_output_dir=eval_path,
                groups=groups,
                pipeline_name=pipeline_name,
            )
            print(f"\n✓ Dashboard saved to: {report_path.absolute()}")
            return 0
        except Exception as e:
            import traceback

            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1


def main() -> int:
    """Main entry point."""
    cli = AnalysisCLI()
    result = fire.Fire(cli)
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
