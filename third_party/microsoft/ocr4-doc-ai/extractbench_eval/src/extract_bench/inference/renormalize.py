"""Utility to re-normalize existing raw inference results."""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from extract_bench.inference.pipelines import get_pipeline
from extract_bench.inference.providers.registry import create_provider
from extract_bench.schemas.pipeline_io import RawInferenceResult

console = Console()


def renormalize_results(
    output_dir: Path,
    pipeline_name: str | None = None,
    force: bool = False,
) -> int:
    """
    Re-normalize existing raw inference results.

    This is useful when the normalization logic has changed but you don't want
    to rerun the expensive inference step.

    :param output_dir: Directory containing raw results (.raw.json files)
    :param pipeline_name: Pipeline name (auto-detected from metadata if not provided)
    :param force: Force re-normalization even if normalized results exist
    :return: Exit code (0 for success, non-zero for failure)
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        console.print(f"[red]Error: Output directory does not exist: {output_dir}")
        return 1

    # Try to get pipeline name from metadata
    if pipeline_name is None:
        metadata_path = output_dir / "_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                pipeline_name = metadata.get("pipeline_name")
            except Exception:
                pass

    if pipeline_name is None:
        console.print("[red]Error: Pipeline name not provided and could not be detected from metadata.")
        console.print("[yellow]Please specify --pipeline_name")
        return 1

    try:
        pipeline_spec = get_pipeline(pipeline_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}")
        return 1

    # Create provider
    try:
        provider = create_provider(pipeline_spec)
    except Exception as e:
        console.print(f"[red]Error creating provider: {e}")
        return 1

    # Find all raw result files
    raw_files = [path for path in output_dir.rglob("*.raw.json") if not path.name.endswith(".error.raw.json")]
    if not raw_files:
        console.print(f"[yellow]No raw result files found in {output_dir}")
        return 0

    console.print(f"[green]Found {len(raw_files)} raw result files")
    console.print(f"[cyan]Pipeline: {pipeline_name}")
    console.print(f"[cyan]Provider: {provider.__class__.__name__}")

    # Process each raw file
    success_count = 0
    error_count = 0
    skipped_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Re-normalizing results...", total=len(raw_files))

        for raw_file in raw_files:
            # Determine normalized file path
            # Replace .raw.json with .result.json
            if raw_file.name.endswith(".raw.json"):
                normalized_file = raw_file.with_name(raw_file.name.replace(".raw.json", ".result.json"))
            else:
                # Fallback: just replace .json with .result.json
                normalized_file = raw_file.with_suffix(".result.json")

            # Check if already normalized (unless force)
            if not force and normalized_file.exists():
                try:
                    # Verify it's valid
                    with open(normalized_file) as f:
                        data = json.load(f)
                    if "request" in data and "output" in data:
                        skipped_count += 1
                        progress.update(task, advance=1)
                        continue
                except Exception:
                    # Invalid file, re-normalize
                    pass

            try:
                # Load raw result
                with open(raw_file) as f:
                    raw_data = json.load(f)
                raw_result = RawInferenceResult.model_validate(raw_data)

                # Normalize
                normalized_result = provider.normalize(raw_result)

                # Save normalized result
                normalized_file.parent.mkdir(parents=True, exist_ok=True)
                with open(normalized_file, "w") as f:
                    f.write(normalized_result.model_dump_json(indent=2))

                success_count += 1
            except Exception as e:
                error_count += 1
                console.print(f"[red]Error processing {raw_file.name}: {e}", style="dim")

            progress.update(task, advance=1)

    # Summary
    console.print("\n[bold]Re-normalization Summary:")
    console.print(f"  [green]Success: {success_count}")
    console.print(f"  [yellow]Skipped: {skipped_count}")
    console.print(f"  [red]Errors: {error_count}")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-normalize existing raw inference results")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing raw results (.raw.json files)",
    )
    parser.add_argument(
        "--pipeline_name",
        type=str,
        help="Pipeline name (auto-detected from metadata if not provided)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-normalization even if normalized results exist",
    )

    args = parser.parse_args()
    sys.exit(renormalize_results(args.output_dir, args.pipeline_name, args.force))
