"""Command-line interface for running end-to-end pipeline benchmarks."""

import sys
import tempfile
from pathlib import Path

import fire

from extract_bench.analysis.aggregation_report import generate_aggregation_report
from extract_bench.analysis.cli import AnalysisCLI
from extract_bench.analysis.leaderboard_report import generate_leaderboard_report
from extract_bench.data.download import default_data_dir, download_dataset, is_dataset_ready
from extract_bench.evaluation.cli import EvaluationCLI
from extract_bench.inference.cli import InferenceCLI
from extract_bench.utils.browser import open_browser

# Shared inference groups: multiple eval categories share one inference dir.
# Maps inference dir name -> list of eval categories.
_SHARED_EVAL_GROUPS = {
    "text": ["text_content", "text_formatting"],
}


def _discover_groups(pipeline_output_dir: Path, dataset_dir: Path | None = None) -> list[str]:
    """Discover evaluation groups from the dataset and inference result files.

    Scans subdirectories of pipeline_output_dir for .result.json files.
    Expands shared inference directories into their eval categories.
    Returns sorted list of group names.

    ``dataset_dir`` is the authority on which groups *should* have been scored.
    Discovering from results alone hides a total failure: a category whose
    documents all errored writes no result file, so it never becomes a group,
    never gets a report, and drops out of the leaderboard as "no data" instead
    of zero. Partial failures within a surviving category already score zero,
    so results-only discovery rewards failing a whole category over failing
    most of one.
    """
    inference_dirs: set[str] = set()
    for result_file in pipeline_output_dir.rglob("*.result.json"):
        parent = result_file.parent
        if parent != pipeline_output_dir:
            inference_dirs.add(parent.name)

    if dataset_dir is not None:
        for test_case in dataset_dir.rglob("*.test.json"):
            parent = test_case.parent
            if parent != dataset_dir:
                inference_dirs.add(parent.name)

    groups: set[str] = set()
    for d in inference_dirs:
        if d in _SHARED_EVAL_GROUPS:
            groups.update(_SHARED_EVAL_GROUPS[d])
        else:
            groups.add(d)
    return sorted(groups)


class PipelineCLI:
    """Command-line interface for running end-to-end benchmarks."""

    def run(
        self,
        pipeline: str,
        input_dir: str | Path | None = None,
        file: str | Path | None = None,
        output_dir: str | Path | None = None,
        max_concurrent: int = 20,
        force: bool = False,
        verbose: bool = False,
        group: str | None = None,
        tags: str | tuple[str, ...] | list[str] | None = None,
        open_report: bool = True,
        skip_inference: bool = False,
        test: bool = False,
    ) -> int:
        """
        Run the full benchmark pipeline: inference -> evaluation -> report -> open browser.

        This command chains together: inference -> evaluation -> report generation.

        Args:
            pipeline: Pipeline name (e.g., 'llamaextract_agentic', 'llamaextract_agentic_plus')
            input_dir: Directory containing test cases/PDFs (default: ./data)
            file: Single file to run (PDF/image). Will use its .test.json if present.
            output_dir: Directory to save results (default: ./output)
            max_concurrent: Maximum concurrent inference requests (default: 20)
            force: Force regeneration even if results already exist (default: False)
            verbose: Enable verbose output (default: False)
            group: Optional group name to filter test cases (e.g., 'chart')
            tags: Tags for this run - comma-separated string or list
            open_report: Open the HTML report in browser when done (default: True)
            skip_inference: Skip inference step, only run evaluation and report (default: False)
            test: Download and run on the small test dataset (6 documents: 3 short, 2 medium, 1 long)

        Returns:
            Exit code (0 for success, non-zero for failure)

        Example:
            extract-bench run llamaextract_agentic
            extract-bench run llamaextract_agentic_plus --max_concurrent 10
            extract-bench run llamaextract_agentic --skip_inference
            extract-bench run llamaextract_agentic --test
        """
        try:
            # Handle single file mode
            if file is not None:
                return self._run_single_file(
                    pipeline=pipeline,
                    file_path=Path(file),
                    output_dir=Path(output_dir) if output_dir else Path("./output"),
                    force=force,
                    verbose=verbose,
                    tags=tags,
                    open_report=open_report,
                    skip_inference=skip_inference,
                )

            # Default input_dir based on --test (./data/test vs ./data) so
            # the test subset doesn't silently get masked by an existing full
            # dataset at ./data, and so the two coexist without overlay.
            # When the user passes --input_dir explicitly we treat that as a
            # custom dataset and skip the public-dataset readiness check /
            # auto-download — otherwise running on a custom dataset would
            # silently scribble HuggingFace files into the user's directory.
            input_dir_explicit = input_dir is not None
            if input_dir is None:
                input_dir = default_data_dir(test=test)

            input_path = Path(input_dir)
            output_base = Path(output_dir) if output_dir else Path("./output")
            pipeline_output_dir = output_base / pipeline

            # Auto-download dataset only when using the default location.
            if not input_dir_explicit and not is_dataset_ready(input_path):
                label = "test dataset" if test else "dataset"
                print(f"{label.capitalize()} not found at {input_path}, downloading from HuggingFace...")
                try:
                    download_dataset(data_dir=input_path, test=test)
                except Exception as e:
                    print(f"Error downloading dataset: {e}", file=sys.stderr)
                    return 1

            # Step 1: Inference
            if not skip_inference:
                print("\n" + "=" * 60)
                print("Step 1/3: Running Inference")
                print("=" * 60 + "\n")

                inference_cli = InferenceCLI()
                exit_code = inference_cli.run(
                    pipeline=pipeline,
                    input_dir=input_path,
                    output_dir=output_base,
                    max_concurrent=max_concurrent,
                    force=force,
                    verbose=verbose,
                    group=group,
                    tags=tags,
                    force_exit_on_completion=False,
                )

                if exit_code != 0:
                    print(f"\nInference failed with exit code {exit_code}", file=sys.stderr)
                    return exit_code
            else:
                print("\n" + "=" * 60)
                print("Step 1/3: Skipping Inference (--skip_inference)")
                print("=" * 60 + "\n")

                if not pipeline_output_dir.exists():
                    print(
                        f"Error: Output directory does not exist: {pipeline_output_dir}",
                        file=sys.stderr,
                    )
                    print("Cannot skip inference without existing results.", file=sys.stderr)
                    return 1

            # Determine if we run per-category or single evaluation
            if group is not None:
                # Single-group mode: unchanged behavior
                return self._run_evaluation_and_report(
                    pipeline_output_dir=pipeline_output_dir,
                    input_path=input_path,
                    verbose=verbose,
                    force=force,
                    group=group,
                    open_report=open_report,
                )
            else:
                # Multi-group mode: per-category evaluation + aggregation dashboard
                return self._run_multi_group_evaluation(
                    pipeline_output_dir=pipeline_output_dir,
                    input_path=input_path,
                    pipeline_name=pipeline,
                    verbose=verbose,
                    force=force,
                    open_report=open_report,
                )

        except KeyboardInterrupt:
            print("\n\nInterrupted by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    def _run_evaluation_and_report(
        self,
        pipeline_output_dir: Path,
        input_path: Path,
        verbose: bool,
        force: bool,
        group: str | None = None,
        open_report: bool = True,
        report_dir: Path | None = None,
    ) -> int:
        """Run evaluation and generate report for a single group or all results.

        Args:
            pipeline_output_dir: Directory containing inference results.
            input_path: Directory containing test cases.
            verbose: Enable verbose output.
            force: Force re-evaluation.
            group: Optional group filter.
            open_report: Open report in browser.
            report_dir: Directory for report output (default: pipeline_output_dir).
        """
        actual_report_dir = report_dir or pipeline_output_dir

        # Step 2: Evaluation
        print("\n" + "=" * 60)
        group_label = f" [{group}]" if group else ""
        print(f"Step 2/3: Running Evaluation{group_label}")
        print("=" * 60 + "\n")

        evaluation_cli = EvaluationCLI()
        exit_code = evaluation_cli.run(
            output_dir=pipeline_output_dir,
            test_cases_dir=input_path,
            verbose=verbose,
            force=force,
            group=group,
            report_dir=str(actual_report_dir),
        )

        if exit_code != 0:
            print(f"\nEvaluation failed with exit code {exit_code}", file=sys.stderr)
            return exit_code

        # Step 3: Generate detailed report
        print("\n" + "=" * 60)
        print(f"Step 3/3: Generating Detailed Report{group_label}")
        print("=" * 60 + "\n")

        # Infer pipeline name from output dir
        inferred_pipeline_name = pipeline_output_dir.name

        analysis_cli = AnalysisCLI()
        exit_code = analysis_cli.generate_report(
            evaluation_dir=actual_report_dir,
            test_cases_dir=input_path,
            output_dir=pipeline_output_dir,
            pipeline_name=inferred_pipeline_name,
            group=group,
        )

        if exit_code != 0:
            print(f"\nReport generation failed with exit code {exit_code}", file=sys.stderr)
            return exit_code

        # Open report in browser
        report_path = actual_report_dir / "_evaluation_report_detailed.html"
        if open_report and report_path.exists():
            print("\n" + "=" * 60)
            print("Evaluation Report")
            print("=" * 60)
            report_url = report_path.absolute().as_uri()
            print(f"\nOpen in browser: {report_url}")
            open_browser(report_url)

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nResults: {pipeline_output_dir}")
        print(f"Report:  {report_path}")

        return 0

    def _run_multi_group_evaluation(
        self,
        pipeline_output_dir: Path,
        input_path: Path,
        pipeline_name: str,
        verbose: bool,
        force: bool,
        open_report: bool = True,
    ) -> int:
        """Run per-category evaluation and generate aggregation dashboard.

        Discovers groups from the dataset and inference results, runs evaluation
        per group, generates per-group detailed reports, then creates an
        aggregation dashboard.
        """
        groups = _discover_groups(pipeline_output_dir, input_path)
        if not groups:
            # No groups found - fall back to single evaluation
            print("No category groups found, running single evaluation")
            return self._run_evaluation_and_report(
                pipeline_output_dir=pipeline_output_dir,
                input_path=input_path,
                verbose=verbose,
                force=force,
                open_report=open_report,
            )

        if len(groups) == 1:
            # Single group - run as single evaluation with report at pipeline root
            print(f"Single group found: {groups[0]}")
            return self._run_evaluation_and_report(
                pipeline_output_dir=pipeline_output_dir,
                input_path=input_path,
                verbose=verbose,
                force=force,
                group=groups[0],
                open_report=open_report,
            )

        print(f"\nDiscovered {len(groups)} categories: {', '.join(groups)}")

        # Reverse lookup: eval group -> inference dir
        _SHARED_INFERENCE_GROUPS = {eg: ig for ig, egs in _SHARED_EVAL_GROUPS.items() for eg in egs}

        # Run evaluation per category
        for i, g in enumerate(groups, 1):
            print("\n" + "=" * 60)
            print(f"Category {i}/{len(groups)}: {g}")
            print("=" * 60)

            # Reports go under the eval group name (e.g., text_content/)
            group_report_dir = pipeline_output_dir / g

            evaluation_cli = EvaluationCLI()
            exit_code = evaluation_cli.run(
                output_dir=pipeline_output_dir,
                test_cases_dir=input_path,
                verbose=verbose,
                force=force,
                group=g,
                report_dir=str(group_report_dir),
            )

            if exit_code != 0:
                print(f"\nEvaluation failed for group '{g}' with exit code {exit_code}", file=sys.stderr)
                # Continue with other groups

            # Generate detailed report for this group
            analysis_cli = AnalysisCLI()
            exit_code = analysis_cli.generate_report(
                evaluation_dir=group_report_dir,
                test_cases_dir=input_path,
                output_dir=pipeline_output_dir,
                pipeline_name=pipeline_name,
                group=g,
            )

            if exit_code != 0:
                print(f"\nReport generation failed for group '{g}'", file=sys.stderr)

        # Generate aggregation dashboard
        print("\n" + "=" * 60)
        print("Generating Aggregation Dashboard")
        print("=" * 60 + "\n")

        dashboard_path = generate_aggregation_report(
            pipeline_output_dir=pipeline_output_dir,
            groups=groups,
            pipeline_name=pipeline_name,
        )

        print(f"Dashboard: {dashboard_path.absolute()}")
        for g in groups:
            detail_path = pipeline_output_dir / g / "_evaluation_report_detailed.html"
            if detail_path.exists():
                print(f"  {g}: {detail_path.absolute()}")

        # Generate leaderboard across all pipelines in the output directory
        output_base = pipeline_output_dir.parent
        print("\n" + "=" * 60)
        print("Generating Leaderboard")
        print("=" * 60 + "\n")

        try:
            leaderboard_path = generate_leaderboard_report(output_dir=output_base)
            print(f"Leaderboard: {leaderboard_path.absolute()}")
        except Exception as e:
            # Non-fatal: leaderboard requires at least one pipeline with results
            print(f"Leaderboard generation skipped: {e}")

        # Open dashboard in browser
        if open_report and dashboard_path.exists():
            dashboard_url = dashboard_path.absolute().as_uri()
            print(f"\nOpen in browser: {dashboard_url}")
            open_browser(dashboard_url)

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nResults: {pipeline_output_dir}")
        print("\nTo view reports with PDF rendering, run:")
        print(f"  uv run extract-bench serve {pipeline_output_dir}")

        return 0

    def _run_single_file(
        self,
        pipeline: str,
        file_path: Path,
        output_dir: Path,
        force: bool,
        verbose: bool,
        tags: str | tuple[str, ...] | list[str] | None,
        open_report: bool,
        skip_inference: bool,
    ) -> int:
        """Run pipeline on a single file by creating a temporary directory structure."""
        import shutil

        file_path = file_path.resolve()

        if not file_path.exists():
            print(f"Error: File does not exist: {file_path}", file=sys.stderr)
            return 1

        # Check for .test.json file
        test_json_path = file_path.parent / f"{file_path.stem}.test.json"
        has_test_json = test_json_path.exists()

        print(f"\nRunning single file: {file_path}")
        if has_test_json:
            print(f"Using test config: {test_json_path}")

        # Create a temporary directory with the expected structure
        # Structure: temp_dir/group/file.pdf + file.test.json
        with tempfile.TemporaryDirectory(prefix="bench_single_") as temp_dir:
            temp_path = Path(temp_dir)
            group_dir = temp_path / "single"
            group_dir.mkdir()

            # Symlink the file (or copy if symlinks not supported)
            temp_file = group_dir / file_path.name
            try:
                temp_file.symlink_to(file_path)
            except OSError:
                shutil.copy2(file_path, temp_file)

            # Symlink/copy the test.json if it exists
            if has_test_json:
                temp_test_json = group_dir / f"{file_path.stem}.test.json"
                try:
                    temp_test_json.symlink_to(test_json_path)
                except OSError:
                    shutil.copy2(test_json_path, temp_test_json)

            # Now run the normal pipeline with this temp directory
            pipeline_output_dir = output_dir / pipeline

            # Step 1: Inference
            if not skip_inference:
                print("\n" + "=" * 60)
                print("Step 1/3: Running Inference")
                print("=" * 60 + "\n")

                inference_cli = InferenceCLI()
                exit_code = inference_cli.run(
                    pipeline=pipeline,
                    input_dir=temp_path,
                    output_dir=output_dir,
                    max_concurrent=1,  # Single file, no need for concurrency
                    force=force,
                    verbose=verbose,
                    tags=tags,
                    force_exit_on_completion=False,
                )

                if exit_code != 0:
                    print(f"\nInference failed with exit code {exit_code}", file=sys.stderr)
                    return exit_code
            else:
                print("\n" + "=" * 60)
                print("Step 1/3: Skipping Inference (--skip_inference)")
                print("=" * 60 + "\n")

                if not pipeline_output_dir.exists():
                    print(
                        f"Error: Output directory does not exist: {pipeline_output_dir}",
                        file=sys.stderr,
                    )
                    print("Cannot skip inference without existing results.", file=sys.stderr)
                    return 1

            # Step 2: Evaluation
            print("\n" + "=" * 60)
            print("Step 2/3: Running Evaluation")
            print("=" * 60 + "\n")

            evaluation_cli = EvaluationCLI()
            exit_code = evaluation_cli.run(
                output_dir=pipeline_output_dir,
                test_cases_dir=temp_path,
                verbose=verbose,
                force=force,
            )

            if exit_code != 0:
                print(f"\nEvaluation failed with exit code {exit_code}", file=sys.stderr)
                return exit_code

            # Step 3: Generate detailed report
            print("\n" + "=" * 60)
            print("Step 3/3: Generating Detailed Report")
            print("=" * 60 + "\n")

            analysis_cli = AnalysisCLI()
            exit_code = analysis_cli.generate_report(
                evaluation_dir=pipeline_output_dir,
                test_cases_dir=temp_path,
            )

            if exit_code != 0:
                print(f"\nReport generation failed with exit code {exit_code}", file=sys.stderr)
                return exit_code

            # Open report in browser
            report_path = pipeline_output_dir / "_evaluation_report_detailed.html"
            if open_report and report_path.exists():
                print("\n" + "=" * 60)
                print("Evaluation Report")
                print("=" * 60)
                report_url = report_path.absolute().as_uri()
                print(f"\nOpen in browser: {report_url}")
                open_browser(report_url)

            print("\n" + "=" * 60)
            print("Pipeline Complete!")
            print("=" * 60)
            print(f"\nResults: {pipeline_output_dir}")
            print(f"Report:  {report_path}")

            return 0


def main() -> int:
    """Main entry point."""
    cli = PipelineCLI()
    result = fire.Fire(cli)
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
