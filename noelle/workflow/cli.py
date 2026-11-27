"""Command-line interface for workflow execution, batch evaluation, and test synthesis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from noelle.evaluation.generators.tts_generator import generate_tts_suite
from noelle.services.cost_advisor import DecisionAdvisor
from noelle.workflow.graph import PrivacyMode, WorkflowGraph
from noelle.workflow.runner import WorkflowRunner


def add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    """Register ``noelle workflow`` subcommands."""
    sub = parser.add_subparsers(dest="workflow_command", metavar="{run,test-gen,advisor}")

    # run command
    run_cmd = sub.add_parser("run", help="execute a workflow manifest against an input or dataset")
    run_cmd.add_argument("manifest", help="path to workflow JSON or YAML file")
    run_cmd.add_argument("--dataset", default=None, help="path to test dataset JSONL file")
    run_cmd.add_argument("--input", "-i", default=None, help="single inline input string")
    run_cmd.add_argument("--output", "-o", default=None, help="save results to JSON file")
    run_cmd.add_argument("--privacy-mode", choices=["local", "hybrid", "cloud"], default=None,
                         help="override privacy mode")
    run_cmd.add_argument("--allow-cloud", action="store_true",
                         help="consent to sending data to configured cloud providers in this run")

    # test-gen command
    tg_cmd = sub.add_parser("test-gen", help="generate adversarial test suites (TTS, STT)")
    tg_cmd.add_argument("--modality", choices=["tts", "stt"], default="tts")
    tg_cmd.add_argument("--category", action="append", default=None, help="category filter (repeatable)")
    tg_cmd.add_argument("--count", type=int, default=10, help="number of test cases to generate")
    tg_cmd.add_argument("--lang", default="id", help="language target (default: id)")
    tg_cmd.add_argument("--output", "-o", default=None, help="save test cases to JSONL")

    # advisor command
    adv_cmd = sub.add_parser("advisor", help="calculate Cloud vs Local TCO and break-even point")
    adv_cmd.add_argument("--queries", type=int, default=500_000, help="projected monthly queries")
    adv_cmd.add_argument("--input-tokens", type=int, default=1200, help="avg input tokens per query")
    adv_cmd.add_argument("--output-tokens", type=int, default=350, help="avg output tokens per query")
    adv_cmd.add_argument("--model", default="gpt-4o", help="cloud model for comparison (default: gpt-4o)")


def main(args: argparse.Namespace) -> int:
    """Dispatch ``noelle workflow <subcommand>``."""
    cmd = getattr(args, "workflow_command", None)

    if cmd == "run":
        return _run_workflow_cli(args)
    elif cmd == "test-gen":
        return _run_test_gen_cli(args)
    elif cmd == "advisor":
        return _run_advisor_cli(args)

    print("Usage: noelle workflow {run,test-gen,advisor} [options]", file=sys.stderr)
    return 1


def _run_workflow_cli(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Error: manifest file '{manifest_path}' does not exist.", file=sys.stderr)
        return 1

    content = manifest_path.read_text(encoding="utf-8")
    try:
        graph = WorkflowGraph.from_json(content)
    except Exception:
        # Fallback to YAML if yaml is installed
        try:
            import yaml
            data = yaml.safe_load(content)
            graph = WorkflowGraph.from_dict(data)
        except Exception as exc:
            print(f"Error parsing manifest as JSON/YAML: {exc}", file=sys.stderr)
            return 1

    if args.privacy_mode:
        graph.privacy_mode = PrivacyMode(args.privacy_mode)

    if any(node.provider_id == "local" for node in graph.nodes.values()):
        # CLI runs have no desktop runtime: spawn one for this process so a local node is
        # not rejected with WORKER_UNAVAILABLE.
        from noelle.providers.local_provider import LocalProvider
        from noelle.providers.registry import register_provider

        register_provider(LocalProvider(auto_spawn=True))

    runner = WorkflowRunner()
    consent = bool(getattr(args, "allow_cloud", False))

    # Batch dataset mode
    if args.dataset:
        ds_path = Path(args.dataset)
        if not ds_path.is_file():
            print(f"Error: dataset file '{ds_path}' not found.", file=sys.stderr)
            return 1

        print(f"Running workflow '{graph.name}' over dataset: {ds_path}...")
        records: list[dict] = []
        with ds_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    try:
                        res = runner.execute(graph, initial_inputs={"in_1": item.get("text", "")},
                                             consent_given=consent)
                    except PermissionError as exc:
                        print(f"Error: {exc}\nRe-run with --allow-cloud to consent.", file=sys.stderr)
                        return 1
                    records.append(res.to_dict())

        total_cost = sum(r["total_cost_usd"] for r in records)
        avg_latency = sum(r["total_latency_ms"] for r in records) / max(1, len(records))
        print(f"Completed {len(records)} items. Avg Latency: {avg_latency:.1f} ms, Total Cost: ${total_cost:.6f}")

        if args.output:
            Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"Saved results to {args.output}")
        return 0

    # Single execution mode
    initial_inputs = {}
    if args.input:
        initial_inputs["in_1"] = args.input

    try:
        res = runner.execute(graph, initial_inputs=initial_inputs, consent_given=consent)
    except PermissionError as exc:
        print(f"Error: {exc}\nRe-run with --allow-cloud to consent.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - zero-egress violations and malformed graphs
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    out_dict = res.to_dict()

    if args.output:
        Path(args.output).write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
        print(f"Saved execution result to {args.output}")
    else:
        print(json.dumps(out_dict, indent=2))

    return 0 if res.is_success else 1


def _run_test_gen_cli(args: argparse.Namespace) -> int:
    cases = generate_tts_suite(categories=args.category, count=args.count, language=args.lang)

    records = [
        {
            "id": c.id,
            "category": c.category,
            "language": c.language,
            "difficulty": c.difficulty,
            "text": c.text,
            "expected_challenges": c.expected_challenges,
        }
        for c in cases
    ]

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(records)} test cases to {args.output}")
    else:
        print(json.dumps(records, indent=2, ensure_ascii=False))

    return 0


def _run_advisor_cli(args: argparse.Namespace) -> int:
    advisor = DecisionAdvisor()
    res = advisor.evaluate(
        projected_monthly_queries=args.queries,
        avg_input_tokens=args.input_tokens,
        avg_output_tokens=args.output_tokens,
        cloud_model=args.model,
    )

    print("\n" + "=" * 65)
    print(" NOELLE HYBRID CLOUD VS LOCAL DECISION ADVISOR")
    print("=" * 65)
    print(f"Projected Workload:     {res.projected_monthly_queries:,} queries / month")
    print(f"Avg Payload Tokens:     {res.avg_input_tokens} input / {res.avg_output_tokens} output")
    print(f"Cloud Comparison Model: {res.cloud_model}")
    print("-" * 65)
    print(f"Cloud API Cost:         ${res.cloud_monthly_cost_usd:,.2f} / month")
    print(f"Local Workstation TCO:  ${res.local_monthly_cost_usd:,.2f} / month (Power + Amortization)")
    print(f"Monthly Net Savings:    ${res.monthly_savings_usd:,.2f} ({res.savings_percentage:.1f}%)")
    print(f"Break-Even Volume:      {res.break_even_queries_per_day:,} queries / day")
    print(f"Payback Parity:         {res.break_even_days:.1f} days")
    print("-" * 65)
    print(f"RECOMMENDED STRATEGY:   {res.recommended_strategy}")
    print(f"Rationale: {res.strategic_rationale}")
    print("=" * 65 + "\n")
    return 0
