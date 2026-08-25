"""Command-line interface for running inference."""

import asyncio
import os
import sys
from collections import defaultdict
from pathlib import Path

import fire  # type: ignore[import-untyped, unused-ignore]
from rich.console import Console
from rich.table import Table

from extract_bench.inference.pipelines import get_pipeline, list_pipelines
from extract_bench.inference.providers.registry import create_provider
from extract_bench.inference.renormalize import renormalize_results
from extract_bench.inference.runner import InferenceRunner
from extract_bench.schemas.product import ProductType
from extract_bench.test_cases import load_test_cases
from extract_bench.test_cases.schema import (
    ExtractTestCase,
    LayoutDetectionTestCase,
    TestCase,
)


def _detect_product_type(test_cases: list[TestCase]) -> ProductType | None:
    """
    Detect product type from test case types.

    :param test_cases: List of loaded test cases
    :return: Detected ProductType or None if unable to detect
    """
    if not test_cases:
        return None

    # Check first test case type to determine product type
    first = test_cases[0]
    if isinstance(first, ExtractTestCase):
        return ProductType.EXTRACT
    if isinstance(first, LayoutDetectionTestCase):
        return ProductType.LAYOUT_DETECTION
    # Default to PARSE for ParseTestCase or unknown types
    return ProductType.PARSE


class InferenceCLI:
    """Command-line interface for running inference on PDFs."""

    def list_pipelines(
        self,
        extract: bool = False,
        parse: bool = False,
        layout: bool = False,
        all: bool = False,  # noqa: A002 - CLI flag name is `--all`
    ) -> None:
        """List available pipelines, grouped by product type.

        ExtractBench is an extraction benchmark, so only extract pipelines are
        listed by default. The parse and layout-detection rosters inherited from
        ParseBench remain fully functional and runnable by name — they are just
        not shown unless asked for. The two-stage extract pipelines use them
        internally as their parse stage.

        Args:
            extract: Show extraction pipelines (the default when no flag is given)
            parse: Show document-parsing pipelines
            layout: Show layout-detection pipelines
            all: Show every registered pipeline

        Example:
            extract-bench pipelines
            extract-bench pipelines --parse
            extract-bench pipelines --all
        """
        if all:
            wanted = None  # no filter
        elif extract or parse or layout:
            wanted = set()
            if extract:
                wanted.add(ProductType.EXTRACT.value)
            if parse:
                wanted.add(ProductType.PARSE.value)
            if layout:
                wanted.add(ProductType.LAYOUT_DETECTION.value)
        else:
            wanted = {ProductType.EXTRACT.value}

        pipelines = list_pipelines()
        if not pipelines:
            print("No pipelines registered.")
            return

        # Group pipelines by product type
        pipelines_by_product: dict[str, list[tuple[str, str]]] = defaultdict(list)
        hidden = 0
        for pipeline_name in pipelines:
            try:
                pipeline_def = get_pipeline(pipeline_name)
                product_type = pipeline_def.product_type.value
            except Exception:
                # If we can't get the pipeline, skip it
                continue
            if wanted is not None and product_type not in wanted:
                hidden += 1
                continue
            pipelines_by_product[product_type].append((pipeline_name, pipeline_def.provider_name))

        if not pipelines_by_product:
            print("No valid pipelines found.")
            return

        console = Console()

        # Sort product types for consistent display
        sorted_products = sorted(pipelines_by_product.keys())

        for product_type in sorted_products:
            # Create a table for this product type
            table = Table(
                title=f"[bold cyan]{product_type.upper()}[/bold cyan]",
                show_header=True,
                header_style="bold magenta",
                box=None,
            )
            table.add_column("Pipeline Name", style="cyan", no_wrap=True)
            table.add_column("Provider", style="green")

            # Sort pipelines within each product type
            pipelines_list = sorted(pipelines_by_product[product_type])
            for pipeline_name, provider_name in pipelines_list:
                table.add_row(pipeline_name, provider_name)

            console.print(table)
            console.print()  # Add spacing between product types

        if hidden:
            console.print(
                f"[dim]{hidden} more pipeline(s) hidden (parse / layout detection). "
                f"Show them with --parse, --layout, or --all.[/dim]\n"
            )

    def run(
        self,
        pipeline: str,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        pipeline_name_override: str | None = None,
        max_concurrent: int = 20,
        save_raw: bool = True,
        save_normalized: bool = True,
        force: bool = False,
        verbose: bool = False,
        no_rich: bool = False,
        group: str | None = None,
        tags: str | tuple[str, ...] | list[str] | None = None,
        per_file_timeout: float | None = None,
        timeout_retries: int = 2,
        force_exit_on_completion: bool = True,
    ) -> int:
        """
        Run inference on a directory, auto-detecting structure and requirements.

        This unified command handles:
        - PARSE with test cases: Structured directory with test.json files
        - PARSE without test cases: Simple directory of PDFs

        Args:
            pipeline: Pipeline name (e.g., 'llamaextract_agentic', 'llamaextract_agentic_plus')
            input_dir: Directory containing files to process (default: ./data)
            output_dir: Directory to save inference results (default: './output')
            pipeline_name_override: Pipeline name override (default: uses pipeline name)
            max_concurrent: Maximum concurrent inference requests (default: 20)
            save_raw: Save raw inference results (default: True)
            save_normalized: Save normalized inference results (default: True)
            force: Force regeneration even if results already exist (default: False)
            verbose: Enable verbose output (default: False)
            no_rich: Disable Rich output for CI environments (default: False)
            group: Optional group name to filter test cases (e.g., 'arxiv_math')
            tags: Tags for this run - comma-separated string or list (e.g., 'nightly,production')
            per_file_timeout: Explicit run-level max seconds per file. Overrides any
                per-pipeline value. None (default) → use the pipeline's per_file_timeout,
                falling back to 600s.
            timeout_retries: Number of retries on per-file timeout (default: 2)
            force_exit_on_completion: Force process exit after inference completes to
                avoid waiting on zombie provider threads (default: True)

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        if input_dir is None:
            input_dir = "./data"
        return self._run_test_cases(
            test_cases_dir=Path(input_dir),
            output_dir=Path(output_dir) if output_dir is not None else Path("./output"),
            pipeline=pipeline,
            pipeline_name_override=pipeline_name_override,
            max_concurrent=max_concurrent,
            save_raw=save_raw,
            save_normalized=save_normalized,
            force=force,
            verbose=verbose,
            no_rich=no_rich,
            group=group,
            tags=tags,
            per_file_timeout=per_file_timeout,
            timeout_retries=timeout_retries,
            force_exit_on_completion=force_exit_on_completion,
        )

    def _run_test_cases(
        self,
        test_cases_dir: Path,
        output_dir: Path,
        pipeline: str,
        pipeline_name_override: str | None,
        max_concurrent: int,
        save_raw: bool,
        save_normalized: bool,
        force: bool,
        verbose: bool,
        no_rich: bool,
        group: str | None,
        tags: str | tuple[str, ...] | list[str] | None,
        per_file_timeout: float | None = None,
        timeout_retries: int = 2,
        force_exit_on_completion: bool = True,
    ) -> int:
        """Internal method to run inference on test cases."""
        try:
            # Get pipeline specification
            try:
                pipeline_spec = get_pipeline(pipeline)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

            # Allow pipeline_name override
            if pipeline_name_override:
                pipeline_spec = pipeline_spec.model_copy(update={"pipeline_name": pipeline_name_override})

            # Append pipeline_name to output_dir
            actual_output_dir = output_dir / pipeline_spec.pipeline_name

            product_type_enum = pipeline_spec.product_type

            # First, try to load test cases without product_type filter to detect type
            # This enables auto-detection for providers that support multiple product types
            try:
                test_cases = load_test_cases(
                    root_dir=test_cases_dir,
                    require_test_json=False,
                    product_type=None,  # Load without filter first
                )
            except ValueError as e:
                print(f"Error loading test cases: {e}", file=sys.stderr)
                return 1

            # Auto-detect product type from test cases
            detected_type = _detect_product_type(test_cases)

            # Check if we should override the pipeline's product type
            # LlamaParse API and local cli2 providers support PARSE -> LAYOUT_DETECTION override.
            if (
                detected_type is not None
                and detected_type != product_type_enum
                and pipeline_spec.provider_name in {"llamaparse"}
                and detected_type == ProductType.LAYOUT_DETECTION
            ):
                print(
                    f"Auto-detected product type: {detected_type.value} (pipeline default: {product_type_enum.value})"
                )
                product_type_enum = detected_type
            elif detected_type == ProductType.EXTRACT and product_type_enum == ProductType.PARSE:
                # Parse pipelines can run over extract datasets when the
                # extract_field rules are used as grounding/evidence tests.
                # Keep the ExtractTestCase objects for file/schema/rule
                # metadata, but run inference as PARSE.
                pass
            elif detected_type != product_type_enum:
                # For other cases, reload with the pipeline's product type filter
                try:
                    test_cases = load_test_cases(
                        root_dir=test_cases_dir,
                        require_test_json=False,
                        product_type=product_type_enum.value,
                    )
                except ValueError as e:
                    print(f"Error loading test cases: {e}", file=sys.stderr)
                    return 1

            # Filter by group if specified
            if group:
                original_count = len(test_cases)
                test_cases = [tc for tc in test_cases if tc.group == group]
                if not test_cases:
                    print(
                        f"No test cases found in group '{group}' in {test_cases_dir}",
                        file=sys.stderr,
                    )
                    return 1
                print(f"Filtered to {len(test_cases)} test cases in group '{group}' (from {original_count} total)")
            else:
                if not test_cases:
                    print(f"No test cases found in {test_cases_dir}", file=sys.stderr)
                    return 1

            # Deduplicate test cases by test_id for inference.
            # e.g. text_content and text_formatting share the same PDFs in docs/text/,
            # so they map to the same test_id — only need to run inference once per file.
            seen_ids: set[str] = set()
            unique_cases: list[TestCase] = []
            for tc in test_cases:
                if tc.test_id not in seen_ids:
                    seen_ids.add(tc.test_id)
                    unique_cases.append(tc)
            if len(unique_cases) < len(test_cases):
                print(
                    f"Deduplicated to {len(unique_cases)} unique files "
                    f"for inference (from {len(test_cases)} test cases)"
                )
            else:
                print(f"Loaded {len(unique_cases)} test cases from {test_cases_dir}")
            test_cases = unique_cases

            # Create provider
            try:
                provider_instance = create_provider(pipeline_spec)
            except Exception as e:
                print(
                    f"Error creating provider '{pipeline_spec.provider_name}': {e}",
                    file=sys.stderr,
                )
                return 1

            # Parse tags - handle both string (comma-separated) and tuple/list (from Fire)
            tags_list: list[str] = []
            if tags:
                if isinstance(tags, (list, tuple)):
                    # Fire may parse comma-separated values as tuple/list
                    tags_list = [str(t).strip() for t in tags if str(t).strip()]
                else:
                    # String with comma-separated values
                    tags_list = [t.strip() for t in tags.split(",") if t.strip()]

            # Create runner
            print(
                f"Creating InferenceRunner with max_concurrent={max_concurrent}, "
                f"per_file_timeout={f'{per_file_timeout}s' if per_file_timeout is not None else 'pipeline/default'}, "
                f"timeout_retries={timeout_retries}"
            )
            runner = InferenceRunner(
                provider=provider_instance,
                pipeline=pipeline_spec,
                output_dir=actual_output_dir,
                max_concurrent=max_concurrent,
                save_raw=save_raw,
                save_normalized=save_normalized,
                force=force,
                use_rich=not (verbose or no_rich),  # Disable Rich if verbose or no_rich flag is set
                tags=tags_list,
                per_file_timeout=per_file_timeout,
                timeout_retries=timeout_retries,
            )

            # Run inference on test cases
            # When max_concurrent is 1, use sync method directly to avoid async overhead
            if max_concurrent == 1:
                summary = runner._run_test_cases_sync(test_cases, product_type_enum, test_cases_dir)
            else:
                summary = asyncio.run(runner.run_test_cases(test_cases, product_type_enum, test_cases_dir))

            # Shutdown the thread pool to prevent zombie threads from blocking exit.
            # When per-file timeouts fire, the underlying threads keep running
            # (Python threads can't be interrupted). Without this, the atexit handler
            # waits forever for those zombie threads to finish.
            runner.shutdown()

            # Print summary
            print("\n" + "=" * 60)
            print("Inference Run Summary")
            print("=" * 60)
            print(f"Total:        {summary.total}")
            print(f"Successful:   {summary.successful}")
            print(f"Failed:       {summary.failed}")
            print(f"Skipped:      {summary.skipped}")
            print(f"Success Rate: {summary.success_rate:.2f}%")
            print(f"Avg Latency:  {summary.avg_latency_ms:.2f}ms")
            print(f"Output Dir:   {actual_output_dir}")
            print("=" * 60)

            if summary.errors:
                errors_file = actual_output_dir / "_errors.json"
                print(f"\n⚠️  {len(summary.errors)} error(s) occurred. See {errors_file}")
                # Print first few errors to console
                print("\nFirst few errors:")
                for i, error in enumerate(summary.errors[:3], 1):
                    example_id = error.get("example_id", "unknown")
                    error_msg = error.get("error", "Unknown error")
                    print(f"\n  {i}. {example_id}: {error_msg}")
                    if error.get("traceback"):
                        traceback_lines = error["traceback"].split("\n")
                        if len(traceback_lines) > 10:
                            print("    Traceback (last 5 lines):")
                            for line in traceback_lines[-5:]:
                                if line.strip():
                                    print(f"      {line}")
                        else:
                            print("    Traceback:")
                            for line in traceback_lines:
                                if line.strip():
                                    print(f"      {line}")
                if len(summary.errors) > 3:
                    remaining = len(summary.errors) - 3
                    print(f"\n  ... and {remaining} more error(s). See {errors_file} for full details.")

            # Return 0 (success) if at least some examples succeeded or all were
            # skipped (results already exist). Only return 1 if there were actual
            # failures with nothing to evaluate.
            exit_code = 0 if (summary.successful > 0 or (summary.failed == 0 and summary.skipped > 0)) else 1

            if force_exit_on_completion:
                # Force-exit to prevent zombie threads from blocking process shutdown.
                # When per-file timeouts fire, the underlying provider threads (e.g.,
                # stuck on Reducto API calls) keep running because Python threads can't
                # be interrupted. The ThreadPoolExecutor atexit handler would wait for
                # these threads forever. Since all results are already saved to disk,
                # os._exit() is safe here.
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(exit_code)
            return exit_code

        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user", file=sys.stderr)

            # Try to save and display partial results
            if "runner" in locals():
                runner.save_partial_results()
                partial_summary = runner.get_current_summary()
                if partial_summary and partial_summary.errors:
                    print(f"\n⚠️  {len(partial_summary.errors)} error(s) before interrupt:")
                    for i, error in enumerate(partial_summary.errors[:5], 1):
                        example_id = error.get("example_id", "unknown")
                        error_msg = error.get("error", "Unknown error")
                        print(f"\n  {i}. {example_id}: {error_msg}")
                        if error.get("traceback"):
                            traceback_lines = error["traceback"].split("\n")
                            print("    Traceback (last 3 lines):")
                            for line in traceback_lines[-3:]:
                                if line.strip():
                                    print(f"      {line}")
                    if len(partial_summary.errors) > 5:
                        remaining = len(partial_summary.errors) - 5
                        errors_file = actual_output_dir / "_errors.json"
                        print(f"\n  ... and {remaining} more. See {errors_file}")

            return 130
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1

    def renormalize(
        self,
        output_dir: str | Path,
        pipeline_name: str | None = None,
        force: bool = False,
    ) -> int:
        """
        Re-normalize existing raw inference results.

        This is useful when the normalization logic has changed but you don't want
        to rerun the expensive inference step.

        Args:
            output_dir: Directory containing raw results (.raw.json files)
            pipeline_name: Pipeline name (auto-detected from metadata if not provided)
            force: Force re-normalization even if normalized results exist

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        return renormalize_results(Path(output_dir), pipeline_name, force)


def main() -> int:
    """Main entry point."""
    cli = InferenceCLI()
    result = fire.Fire(cli)
    # Fire returns the result of the called method
    # If it's an integer (exit code), use it; otherwise default to 0
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
