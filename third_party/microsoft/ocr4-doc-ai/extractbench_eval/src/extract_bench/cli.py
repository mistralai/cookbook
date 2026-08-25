"""Unified command-line interface for extract-bench."""

from __future__ import annotations

import sys
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import fire
from dotenv import load_dotenv

if TYPE_CHECKING:
    # Lazy import/instantiate CLIs to minimize startup time and dependency load.
    from extract_bench.analysis.cli import AnalysisCLI
    from extract_bench.data.cli import DataCLI
    from extract_bench.evaluation.cli import EvaluationCLI
    from extract_bench.inference.cli import InferenceCLI
    from extract_bench.pipeline.cli import PipelineCLI


# Load .env file if it exists (look in current directory and project root)
def _load_env() -> None:
    """Load environment variables from .env file."""
    # Try current directory first, then project root
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)  # Don't override existing env vars
            break


def _resolve_pipeline_dir(name_or_path: str | Path) -> Path:
    """Resolve a pipeline name or path to a directory.

    If the input is an existing directory, use it as-is.
    Otherwise, try ./output/<name>.
    """
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = Path("./output") / p
    if candidate.exists():
        return candidate
    return p  # Return original; caller will handle the error


class BenchCLI:
    """Unified CLI for extract-bench.

    Top-level commands (recommended):
        run          Run end-to-end benchmark pipeline
        download     Download dataset from HuggingFace
        status       Check if dataset is ready
        pipelines    List available pipeline configurations
        compare      Compare two pipeline results
        serve        View reports in browser with PDF support

    Advanced subcommands:
        inference    Run inference only
        evaluation   Run evaluation only
        analysis     Generate reports, dashboards, comparisons
        pipeline     End-to-end pipeline (same as 'run')
        data         Dataset management (same as 'download'/'status')
    """

    # ── Subcommand ──────────────────────────────────────────────────

    @cached_property
    def inference(self) -> InferenceCLI:
        from extract_bench.inference.cli import InferenceCLI

        return InferenceCLI()

    @cached_property
    def evaluation(self) -> EvaluationCLI:
        from extract_bench.evaluation.cli import EvaluationCLI

        return EvaluationCLI()

    @cached_property
    def analysis(self) -> AnalysisCLI:
        from extract_bench.analysis.cli import AnalysisCLI

        return AnalysisCLI()

    @cached_property
    def pipeline(self) -> PipelineCLI:
        from extract_bench.pipeline.cli import PipelineCLI

        return PipelineCLI()

    @cached_property
    def data(self) -> DataCLI:
        from extract_bench.data.cli import DataCLI

        return DataCLI()

    # ── Top-level convenience commands ──────────────────────────────

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
        """Run end-to-end benchmark: inference -> evaluation -> report.

        Args:
            pipeline: Pipeline name (e.g., 'llamaextract_agentic', 'llamaextract_cost_effective')
            input_dir: Directory containing test cases/PDFs (default: ./data)
            file: Single file to run (PDF/image)
            output_dir: Directory to save results (default: ./output)
            max_concurrent: Maximum concurrent inference requests (default: 20)
            force: Force regeneration even if results exist (default: False)
            verbose: Enable verbose output (default: False)
            group: Filter by category (e.g., 'short', 'medium', 'long')
            tags: Tags for this run
            open_report: Open HTML report in browser (default: True)
            skip_inference: Skip inference, only re-evaluate (default: False)
            test: Download and run on the small test dataset (6 documents: 3 short, 2 medium, 1 long)

        Example:
            extract-bench run llamaextract_agentic
            extract-bench run llamaextract_agentic --group short
            extract-bench run llamaextract_agentic --skip_inference
            extract-bench run llamaextract_agentic --test
        """
        return self.pipeline.run(
            pipeline=pipeline,
            input_dir=input_dir,
            file=file,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            force=force,
            verbose=verbose,
            group=group,
            tags=tags,
            open_report=open_report,
            skip_inference=skip_inference,
            test=test,
        )

    def download(
        self,
        data_dir: str | Path | None = None,
        force: bool = False,
        test: bool = False,
    ) -> int:
        """Download the benchmark dataset from HuggingFace.

        Args:
            data_dir: Directory to store dataset (default: ./data)
            force: Force re-download even if data exists
            test: Download the small test dataset (6 documents: 3 short, 2 medium, 1 long)

        Example:
            extract-bench download
            extract-bench download --test
        """
        return self.data.download(data_dir=data_dir, force=force, test=test)

    def status(
        self,
        data_dir: str | Path | None = None,
        test: bool = False,
    ) -> int:
        """Check if the benchmark dataset is downloaded and ready.

        Args:
            data_dir: Data directory to check
                (default: ./data, or ./data/test when --test is set)
            test: Check the small test dataset instead of the full dataset

        Example:
            extract-bench status
            extract-bench status --test
            extract-bench status data/
        """
        return self.data.status(data_dir=data_dir, test=test)

    def pipelines(
        self,
        extract: bool = False,
        parse: bool = False,
        layout: bool = False,
        all: bool = False,  # noqa: A002 - CLI flag name is `--all`
    ) -> None:
        """List available pipelines (extraction pipelines by default).

        The parse and layout-detection pipelines inherited from ParseBench are
        still registered and runnable by name; they are hidden here so the
        default listing shows the extraction roster this benchmark scores.

        Args:
            extract: Show extraction pipelines (default when no flag is given)
            parse: Show document-parsing pipelines
            layout: Show layout-detection pipelines
            all: Show every registered pipeline

        Example:
            extract-bench pipelines
            extract-bench pipelines --parse
            extract-bench pipelines --all
        """
        return self.inference.list_pipelines(extract=extract, parse=parse, layout=layout, all=all)

    def compare(
        self,
        pipeline_a: str | Path,
        pipeline_b: str | Path,
        test_cases_dir: str | Path | None = None,
        output_file: str | Path | None = None,
        pdf_base_url: str | None = None,
    ) -> int:
        """Compare results from two pipelines.

        Pipeline names are auto-resolved to ./output/<name> if the path
        doesn't exist as-is.

        Args:
            pipeline_a: Pipeline A name or directory (e.g., 'llamaextract_agentic')
            pipeline_b: Pipeline B name or directory (e.g., 'llamaextract_cost_effective')
            test_cases_dir: Directory containing test cases (default: auto-detect)
            output_file: Path to save comparison report (default: auto)
            pdf_base_url: Optional base URL/path for PDF preview (default: auto from test_cases_dir)

        Example:
            extract-bench compare llamaextract_agentic llamaextract_cost_effective
            extract-bench compare ./output/llamaextract_agentic ./output/llamaextract_cost_effective
        """
        return self.analysis.compare_pipelines(
            pipeline_a_dir=_resolve_pipeline_dir(pipeline_a),
            pipeline_b_dir=_resolve_pipeline_dir(pipeline_b),
            test_cases_dir=test_cases_dir,
            output_file=output_file,
            pdf_base_url=pdf_base_url,
        )

    def leaderboard(
        self,
        *pipelines: str,
        output_dir: str | Path = "./output",
        output_file: str | Path | None = None,
    ) -> int:
        """Generate a leaderboard comparing all pipelines side-by-side.

        If no pipeline names are given, auto-discovers all pipelines in the
        output directory.

        Args:
            *pipelines: Optional pipeline names to include (e.g., 'llamaextract_agentic')
            output_dir: Parent directory containing pipeline subdirectories
            output_file: Path to save the leaderboard HTML

        Example:
            extract-bench leaderboard
            extract-bench leaderboard llamaextract_agentic llamaextract_cost_effective
        """
        pipeline_list = list(pipelines) if pipelines else None
        return self.analysis.generate_leaderboard(
            output_dir=output_dir,
            pipelines=pipeline_list,
            output_file=output_file,
        )

    def serve(
        self,
        pipeline: str | Path | None = None,
        port: int = 8080,
        root: str | Path = ".",
    ) -> int:
        """Start a local server to view reports with PDF rendering support.

        Pipeline names are auto-resolved to ./output/<name> if the path
        doesn't exist as-is.

        Args:
            pipeline: Pipeline name or directory (e.g., 'llamaextract_agentic')
            port: Port number (default: 8080)
            root: Root directory to serve (default: current directory)

        Example:
            extract-bench serve llamaextract_agentic
            extract-bench serve ./output/llamaextract_agentic
            extract-bench serve
        """
        pipeline_dir = _resolve_pipeline_dir(pipeline) if pipeline else None
        return self.analysis.serve(
            pipeline_dir=pipeline_dir,
            port=port,
            root=root,
        )


def main() -> int:
    """Main entry point for the unified CLI."""
    # Load .env file before any commands run
    _load_env()
    cli = BenchCLI()
    result = fire.Fire(cli)
    # Fire returns the result of the called method
    # If it's an integer (exit code), use it; otherwise default to 0
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
