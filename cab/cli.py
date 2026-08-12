"""CLI for Cognitive Automaton Bench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="cab")
def main():
    """Cognitive Automaton Bench — evaluate emergent cognition in LLMs."""
    pass


@main.command()
@click.option("--model", "-m", required=True, help="Model to benchmark (e.g., gpt-4o, claude-sonnet-4)")
@click.option("--suite", "-s", default="core", help="Benchmark suite: core, full, survival, emergence, social, cognitive")
@click.option("--benchmark", "-b", help="Run a single benchmark by ID (overrides --suite)")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="API key for the LLM provider")
@click.option("--output-dir", "-o", default="results", help="Output directory for results")
@click.option("--base-url", envvar="XIAOMI_BASE_URL", help="Base URL for OpenAI-compatible API")
@click.option("--temperature", "-t", default=0.7, type=float, help="LLM temperature")
@click.option("--max-tokens", default=2048, type=int, help="Max tokens per LLM call")
@click.option("--dry-run", is_flag=True, help="Print benchmarks that would run without executing")
def run(model, suite, benchmark, api_key, base_url, output_dir, temperature, max_tokens, dry_run):
    """Run benchmarks against a model."""
    from cab.benchmarks import BENCHMARKS, SUITES
    from cab.runners import LLMRunner
    from cab.scorer import save_results, aggregate_results

    # Determine which benchmarks to run
    if benchmark:
        if benchmark not in BENCHMARKS:
            console.print(f"[red]Unknown benchmark: {benchmark}[/red]")
            console.print(f"Available: {', '.join(sorted(BENCHMARKS.keys()))}")
            sys.exit(1)
        bench_ids = [benchmark]
    elif suite in SUITES:
        bench_ids = SUITES[suite]
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        console.print(f"Available suites: {', '.join(SUITES.keys())}")
        sys.exit(1)

    if dry_run:
        console.print(f"[bold]Would run {len(bench_ids)} benchmarks against {model}:[/bold]")
        for bid in bench_ids:
            cls = BENCHMARKS[bid]
            b = cls()
            console.print(f"  {bid} ({b.difficulty.value}) — {b.name}")
        return

    console.print(Panel(
        f"[bold cyan]Cognitive Automaton Bench[/bold cyan]\n"
        f"Model: [yellow]{model}[/yellow]\n"
        f"Suite: [yellow]{suite}[/yellow]\n"
        f"Benchmarks: [yellow]{len(bench_ids)}[/yellow]",
        title="🚀 Starting Run",
    ))

    runner = LLMRunner(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for bid in bench_ids:
            cls = BENCHMARKS[bid]
            bench = cls()
            task = progress.add_task(f"Running {bid}...", total=None)
            try:
                result = bench.run(runner, model)
                results.append(result)
                status = "[green]✓[/green]" if result.completed else "[red]✗[/red]"
                progress.update(task, description=(
                    f"{status} {bid} — CAS: {result.cas:.3f} "
                    f"(F:{result.fitness:.2f} E:{result.efficiency:.2f} "
                    f"Em:{result.emergence:.2f} R:{result.robustness:.2f} Fi:{result.fidelity:.2f})"
                ))
            except Exception as e:
                progress.update(task, description=f"[red]✗ {bid} — Error: {e}[/red]")
                from cab.core import BenchmarkResult, BenchmarkCategory, Difficulty
                error_result = BenchmarkResult(
                    benchmark_id=bid,
                    model=model,
                    category=bench.category,
                    difficulty=bench.difficulty,
                    error=str(e),
                )
                results.append(error_result)

    # Summary
    summary = aggregate_results(results)
    filepath = save_results(results, model, output_dir)

    console.print()
    table = Table(title=f"Results — {model}")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", style="yellow", justify="right")
    for dim, score in summary["dimensions"].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        table.add_row(dim.upper(), f"{score:.4f}  {bar}")
    console.print(table)

    console.print(Panel(
        f"[bold green]CAS: {summary['overall_cas']:.4f}[/bold green]\n"
        f"Completed: {summary['completed']}/{summary['count']}\n"
        f"Tokens: {summary['total_tokens']:,}\n"
        f"Results saved to: {filepath}",
        title="📊 Final Score",
    ))


@main.command()
@click.argument("result_files", nargs=-1, required=True)
def compare(result_files):
    """Compare results from multiple model runs."""
    from cab.core import BenchmarkResult, BenchmarkCategory, Difficulty
    from cab.scorer import compare_models

    result_sets = {}
    for filepath in result_files:
        with open(filepath) as f:
            data = json.load(f)
        model = data["model"]
        # Reconstruct BenchmarkResult objects
        results = []
        for b in data["benchmarks"]:
            r = BenchmarkResult(
                benchmark_id=b["benchmark_id"],
                model=b["model"],
                category=BenchmarkCategory(b["category"]),
                difficulty=Difficulty(b["difficulty"]),
                fitness=b["scores"]["fitness"],
                efficiency=b["scores"]["efficiency"],
                emergence=b["scores"]["emergence"],
                robustness=b["scores"]["robustness"],
                fidelity=b["scores"]["fidelity"],
                cas=b["scores"]["cas"],
                total_tokens=b["total_tokens"],
                total_latency_ms=b["total_latency_ms"],
                completed=b["completed"],
            )
            results.append(r)
        result_sets[model] = results

    table_str = compare_models(result_sets)
    console.print(table_str)


@main.command()
@click.option("--suite", "-s", default="full", help="Suite to list")
def list(suite):
    """List available benchmarks."""
    from cab.benchmarks import BENCHMARKS, SUITES

    if suite == "all":
        bench_ids = list(BENCHMARKS.keys())
    elif suite in SUITES:
        bench_ids = SUITES[suite]
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        return

    table = Table(title=f"Benchmarks — {suite} suite")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Category", style="yellow")
    table.add_column("Difficulty", style="green")
    table.add_column("Max Turns", justify="right")

    for bid in bench_ids:
        cls = BENCHMARKS[bid]
        b = cls()
        table.add_row(bid, b.name, b.category.value, b.difficulty.value, str(b.max_turns))

    console.print(table)


@main.command()
@click.argument("result_file")
@click.option("--format", "-f", "fmt", default="markdown", type=click.Choice(["markdown", "json", "html"]))
def report(result_file, fmt):
    """Generate a detailed report from results."""
    with open(result_file) as f:
        data = json.load(f)

    if fmt == "json":
        click.echo(json.dumps(data, indent=2))
        return

    # Markdown report
    model = data["model"]
    summary = data["summary"]

    lines = [
        f"# Cognitive Automaton Bench Report",
        f"",
        f"**Model:** {model}",
        f"**Date:** {data['timestamp']}",
        f"",
        f"## Overall Score",
        f"",
        f"**CAS (Cognitive Automaton Score): {summary['overall_cas']:.4f}**",
        f"",
        f"| Dimension | Score |",
        f"|---|---|",
    ]
    for dim, score in summary["dimensions"].items():
        lines.append(f"| {dim.title()} | {score:.4f} |")

    lines.extend([
        f"",
        f"## Category Breakdown",
        f"",
        f"| Category | CAS |",
        f"|---|---|",
    ])
    for cat, score in summary["by_category"].items():
        lines.append(f"| {cat.title()} | {score:.4f} |")

    lines.extend([
        f"",
        f"## Difficulty Breakdown",
        f"",
        f"| Difficulty | CAS |",
        f"|---|---|",
    ])
    for diff, score in summary["by_difficulty"].items():
        lines.append(f"| {diff.title()} | {score:.4f} |")

    lines.extend([
        f"",
        f"## Individual Benchmarks",
        f"",
        f"| Benchmark | CAS | Fitness | Efficiency | Emergence | Robustness | Fidelity |",
        f"|---|---|---|---|---|---|---|",
    ])
    for b in data["benchmarks"]:
        s = b["scores"]
        status = "✓" if b["completed"] else "✗"
        lines.append(
            f"| {status} {b['benchmark_id']} | {s['cas']:.4f} | {s['fitness']:.4f} | "
            f"{s['efficiency']:.4f} | {s['emergence']:.4f} | {s['robustness']:.4f} | {s['fidelity']:.4f} |"
        )

    lines.extend([
        f"",
        f"## Statistics",
        f"",
        f"- Total tokens: {summary['total_tokens']:,}",
        f"- Total latency: {summary['total_latency_ms']:.0f}ms",
        f"- Benchmarks completed: {summary['completed']}/{summary['count']}",
    ])

    click.echo("\n".join(lines))


if __name__ == "__main__":
    main()
