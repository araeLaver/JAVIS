"""Model version management CLI with auto-training support."""

import argparse
import json
import shutil
import signal
import sys
from datetime import datetime
from pathlib import Path


def list_versions(models_dir: str = "models"):
    """List all model versions."""
    models_path = Path(models_dir)

    if not models_path.exists():
        print("No models directory found.")
        return

    print(f"\n{'Version':<25} {'Created':<20} {'Dataset Size':<15} {'Status'}")
    print("=" * 80)

    versions = []
    for version_dir in sorted(models_path.iterdir(), reverse=True):
        if not version_dir.is_dir():
            continue

        metadata_file = version_dir / "metadata.json"
        adapter_dir = version_dir / "adapter"

        if not adapter_dir.exists():
            continue

        info = {
            "version": version_dir.name,
            "created": "-",
            "dataset_size": "-",
            "status": "ready",
        }

        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                info["created"] = metadata.get("created_at", "-")[:10]
                info["dataset_size"] = str(metadata.get("dataset_size", "-"))

        versions.append(info)
        print(f"{info['version']:<25} {info['created']:<20} {info['dataset_size']:<15} {info['status']}")

    if not versions:
        print("No model versions found.")

    print()


def export_data(
    output_path: str = None,
    feedback_filter: str = None,
    conversations_dir: str = "data/conversations"
):
    """Export conversations as training data."""
    conv_path = Path(conversations_dir)

    if not conv_path.exists():
        print(f"No conversations directory: {conversations_dir}")
        return

    if output_path is None:
        output_path = f"data/training/exported/conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    conversations = []
    total_files = 0
    filtered_out = 0

    for json_file in conv_path.rglob("*.json"):
        total_files += 1
        with open(json_file, 'r', encoding='utf-8') as f:
            conv = json.load(f)

            # Feedback filter
            if feedback_filter and conv.get('feedback') != feedback_filter:
                filtered_out += 1
                continue

            # At least 2 turns (user + assistant)
            if len(conv.get('turns', [])) >= 2:
                conversations.append(conv)

    # Export as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            messages = []
            for turn in conv['turns']:
                messages.append({
                    "role": turn['role'],
                    "content": turn['content']
                })
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')

    print(f"\nExport complete!")
    print(f"  Total files: {total_files}")
    print(f"  Filtered out: {filtered_out}")
    print(f"  Exported: {len(conversations)}")
    print(f"  Output: {output_file}")


def create_version(adapter_path: str, version_name: str = None, models_dir: str = "models"):
    """Create a new version from an adapter."""
    adapter_src = Path(adapter_path)

    if not adapter_src.exists():
        print(f"Adapter not found: {adapter_path}")
        return

    if version_name is None:
        version_name = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    models_path = Path(models_dir)
    version_dir = models_path / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy adapter
    adapter_dest = version_dir / "adapter"
    shutil.copytree(adapter_src, adapter_dest)

    # Create metadata if not exists
    metadata_file = version_dir / "metadata.json"
    if not metadata_file.exists():
        metadata = {
            "version": version_name,
            "created_at": datetime.now().isoformat(),
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
        }

        # Try to get metadata from adapter
        src_metadata = adapter_src / "metadata.json"
        if src_metadata.exists():
            with open(src_metadata, 'r') as f:
                metadata.update(json.load(f))

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nVersion created: {version_name}")
    print(f"  Path: {version_dir}")


def stats(conversations_dir: str = "data/conversations"):
    """Show conversation statistics."""
    conv_path = Path(conversations_dir)

    if not conv_path.exists():
        print("No conversations found.")
        return

    total_convs = 0
    total_turns = 0
    good_feedback = 0
    bad_feedback = 0
    no_feedback = 0

    for json_file in conv_path.rglob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            conv = json.load(f)
            total_convs += 1
            total_turns += len(conv.get('turns', []))

            feedback = conv.get('feedback')
            if feedback == 'good':
                good_feedback += 1
            elif feedback == 'bad':
                bad_feedback += 1
            else:
                no_feedback += 1

    print(f"\n{'='*40}")
    print(f"JAVIS Training Data Statistics")
    print(f"{'='*40}")
    print(f"Total conversations: {total_convs}")
    print(f"Total turns: {total_turns}")
    print(f"")
    print(f"Feedback breakdown:")
    print(f"  [+] Good: {good_feedback}")
    print(f"  [-] Bad: {bad_feedback}")
    print(f"  [ ] None: {no_feedback}")
    print(f"")
    print(f"Ready for training: {good_feedback + no_feedback} conversations")
    print(f"{'='*40}")


def scheduler_command(action: str):
    """Control the training scheduler."""
    from .scheduler import get_scheduler, start_scheduler, stop_scheduler

    scheduler = get_scheduler()

    if action == "start":
        print("Starting scheduler...")
        if start_scheduler():
            status = scheduler.get_status()
            print(f"Scheduler started!")
            print(f"  Cron: {status.cron}")
            print(f"  Timezone: {status.timezone}")
            print(f"  Next run: {status.next_run or 'Not scheduled'}")

            # Keep running until interrupted
            print("\nPress Ctrl+C to stop...")

            def signal_handler(sig, frame):
                print("\nStopping scheduler...")
                stop_scheduler()
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            # Block forever
            signal.pause() if hasattr(signal, 'pause') else input()
        else:
            print("Failed to start scheduler. Check config and logs.")

    elif action == "stop":
        print("Stopping scheduler...")
        if stop_scheduler():
            print("Scheduler stopped.")
        else:
            print("Failed to stop scheduler.")

    elif action == "status":
        status = scheduler.get_status()
        print(f"\n{'='*40}")
        print("Training Scheduler Status")
        print(f"{'='*40}")
        print(f"Enabled: {status.enabled}")
        print(f"Running: {status.running}")
        print(f"Cron: {status.cron}")
        print(f"Timezone: {status.timezone}")
        print(f"Next run: {status.next_run or 'N/A'}")
        print(f"Last run: {status.last_run or 'Never'}")
        print(f"Last result: {status.last_result or 'N/A'}")
        print(f"{'='*40}")

    elif action == "trigger":
        print("Manually triggering training...")
        scheduler.trigger_now()
        print("Training triggered. Check logs for progress.")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: start, stop, status, trigger")


def train_command(provider: str = None, force: bool = False, dry_run: bool = False):
    """Run training pipeline."""
    from .pipeline import TrainingPipeline

    print("Initializing training pipeline...")
    pipeline = TrainingPipeline()

    if dry_run:
        # Just check conditions
        can_train, reason = pipeline.check_conditions()
        data_stats = pipeline.get_data_stats()

        print(f"\n{'='*40}")
        print("Training Pipeline - Dry Run")
        print(f"{'='*40}")
        print(f"Provider: {pipeline.config.provider}")
        print(f"Base model: {pipeline.config.model.base_model}")
        print(f"\nData Statistics:")
        print(f"  Total conversations: {data_stats.total_conversations}")
        print(f"  Good feedback: {data_stats.good_feedback}")
        print(f"  Ready for training: {data_stats.ready_for_training}")
        print(f"\nConditions:")
        print(f"  Can train: {can_train}")
        print(f"  Reason: {reason}")
        print(f"\nTraining config:")
        print(f"  Epochs: {pipeline.config.model.epochs}")
        print(f"  Batch size: {pipeline.config.model.batch_size}")
        print(f"  Learning rate: {pipeline.config.model.learning_rate}")
        print(f"  LoRA rank: {pipeline.config.model.lora_r}")
        print(f"{'='*40}")
        return

    print("Starting training...")
    result = pipeline.run(force=force)

    print(f"\n{'='*40}")
    print("Training Result")
    print(f"{'='*40}")

    if result.skipped:
        print(f"Status: SKIPPED")
        print(f"Reason: {result.skip_reason}")
    elif result.success:
        print(f"Status: SUCCESS")
        print(f"Version: {result.version}")
        print(f"Dataset size: {result.dataset_size}")
        print(f"Duration: {result.duration_seconds:.1f}s")
    else:
        print(f"Status: FAILED")
        print(f"Error: {result.error}")

    print(f"{'='*40}")


def rollback_command(version: str = None):
    """Rollback to a previous model version."""
    from .version_manager import get_version_manager

    vm = get_version_manager()
    current = vm.get_active_version()

    if version:
        # Rollback to specific version
        print(f"Rolling back to version: {version}")
        if vm.activate_version(version):
            print(f"Successfully rolled back from {current or 'none'} to {version}")
        else:
            print(f"Failed to rollback. Version {version} not found.")
    else:
        # Rollback to previous version
        print("Rolling back to previous version...")
        previous = vm.rollback()
        if previous:
            print(f"Successfully rolled back from {current or 'none'} to {previous}")
        else:
            print("No previous version to rollback to.")


def active_command():
    """Show currently active model version."""
    from .version_manager import get_version_manager

    vm = get_version_manager()
    active = vm.get_active_version()

    if active:
        info = vm.get_version_info(active)
        print(f"\n{'='*40}")
        print("Active Model Version")
        print(f"{'='*40}")
        print(f"Version: {active}")
        if info:
            print(f"Created: {info.created_at}")
            print(f"Base model: {info.base_model}")
            print(f"Dataset size: {info.dataset_size}")
        adapter_path = vm.get_active_adapter_path()
        print(f"Adapter path: {adapter_path}")
        print(f"{'='*40}")
    else:
        print("No active model version.")


# Phase 5 Commands

def quality_command(action: str, min_score: float = 3.5, output: str = None):
    """Data quality management commands."""
    from .data_quality import get_quality_scorer

    scorer = get_quality_scorer()

    if action == "stats":
        stats = scorer.get_statistics()
        print(f"\n{'='*40}")
        print("Data Quality Statistics")
        print(f"{'='*40}")
        print(f"Total conversations: {stats['total']}")
        print(f"High quality (>=4.0): {stats['high_quality']}")
        print(f"Medium quality (3.0-4.0): {stats['medium_quality']}")
        print(f"Low quality (<3.0): {stats['low_quality']}")
        print(f"Average score: {stats['average_score']}")
        if stats['total'] > 0:
            print(f"Score range: {stats['min_score']} - {stats['max_score']}")
        print(f"{'='*40}")

    elif action == "score-all":
        print("Scoring all conversations...")
        scores = scorer.score_all_conversations()
        print(f"\nScored {len(scores)} conversations")
        for score in sorted(scores, key=lambda x: x.overall_score, reverse=True)[:10]:
            print(f"  {score.conversation_id}: {score.overall_score}")
        if len(scores) > 10:
            print(f"  ... and {len(scores) - 10} more")

    elif action == "export":
        output_path = Path(output) if output else None
        path = scorer.export_quality_report(output_path)
        print(f"Exported quality report to: {path}")

    elif action == "high-quality":
        ids = scorer.get_high_quality_ids(min_score)
        print(f"\nHigh quality conversations (score >= {min_score}):")
        print(f"{'='*40}")
        for cid in ids:
            print(f"  {cid}")
        print(f"\nTotal: {len(ids)}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: stats, score-all, export, high-quality")


def metrics_command(action: str, version: str = None, version_a: str = None, version_b: str = None, days: int = 7):
    """Performance metrics commands."""
    from .metrics import get_metrics_collector

    collector = get_metrics_collector()

    if action == "show":
        if not version:
            print("Error: --version required")
            return

        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=days)

        metrics = collector.get_metrics(version, start, end)
        print(f"\n{'='*50}")
        print(f"Performance Metrics: {version}")
        print(f"{'='*50}")
        print(f"Period: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        print(f"Total requests: {metrics.total_requests}")
        print(f"Average latency: {metrics.avg_latency_ms:.2f} ms")
        print(f"P95 latency: {metrics.p95_latency_ms:.2f} ms")
        print(f"Positive feedback rate: {metrics.positive_feedback_rate*100:.1f}%")
        print(f"Negative feedback rate: {metrics.negative_feedback_rate*100:.1f}%")
        print(f"Error rate: {metrics.error_rate*100:.1f}%")
        print(f"{'='*50}")

    elif action == "compare":
        if not version_a or not version_b:
            print("Error: --version-a and --version-b required")
            return

        comparison = collector.compare_versions(version_a, version_b, days)
        print(f"\n{'='*60}")
        print(f"Version Comparison ({days} days)")
        print(f"{'='*60}")
        print(f"\n{'Metric':<25} {'Version A':<15} {'Version B':<15} {'Change'}")
        print("-" * 60)

        va = comparison["version_a"]
        vb = comparison["version_b"]
        ch = comparison["comparison"]

        print(f"{'Version':<25} {va['version']:<15} {vb['version']:<15}")
        print(f"{'Requests':<25} {va['requests']:<15} {vb['requests']:<15}")
        print(f"{'Avg Latency (ms)':<25} {va['avg_latency_ms']:<15} {vb['avg_latency_ms']:<15} {ch['latency_change_pct']:+.1f}%")
        print(f"{'Positive Rate':<25} {va['positive_feedback_rate']*100:.1f}%{'':<9} {vb['positive_feedback_rate']*100:.1f}%{'':<9} {ch['feedback_change_pct']:+.1f}%")
        print(f"{'Error Rate':<25} {va['error_rate']*100:.1f}%{'':<10} {vb['error_rate']*100:.1f}%{'':<10} {ch['error_change_pct']:+.1f}%")
        print(f"{'='*60}")

    elif action == "dashboard":
        data = collector.get_dashboard_data()
        print(f"\n{'='*60}")
        print("Metrics Dashboard (Last 7 Days)")
        print(f"{'='*60}")
        print(f"Total requests: {data['total_requests']}")
        print(f"\n{'Version':<20} {'Requests':<12} {'Latency':<12} {'Positive':<10} {'Errors'}")
        print("-" * 60)
        for v in data["versions"]:
            print(f"{v['version']:<20} {v['requests']:<12} {v['avg_latency_ms']:<12.1f} {v['positive_rate']:<10.1f}% {v['error_rate']:.1f}%")
        print(f"{'='*60}")

    elif action == "cleanup":
        removed = collector.cleanup_old_metrics()
        print(f"Cleaned up {removed} old metrics files")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: show, compare, dashboard, cleanup")


def abtest_command(action: str, version_a: str = None, version_b: str = None, test_id: str = None, split: float = 0.5, winner: str = None):
    """A/B testing commands."""
    from .ab_testing import get_ab_test_manager

    manager = get_ab_test_manager()

    if action == "create":
        if not version_a or not version_b:
            print("Error: --version-a and --version-b required")
            return

        test_id = manager.create_test(version_a, version_b, split)
        print(f"\n{'='*50}")
        print("A/B Test Created")
        print(f"{'='*50}")
        print(f"Test ID: {test_id}")
        print(f"Version A (control): {version_a}")
        print(f"Version B (treatment): {version_b}")
        print(f"Traffic split: {split*100:.0f}% to version B")
        print(f"{'='*50}")

    elif action == "list":
        tests = manager.list_tests()
        print(f"\n{'='*70}")
        print("A/B Tests")
        print(f"{'='*70}")
        print(f"{'Test ID':<35} {'Status':<12} {'Start Date'}")
        print("-" * 70)
        for test in tests:
            print(f"{test.test_id:<35} {test.status:<12} {test.start_time.strftime('%Y-%m-%d')}")
        if not tests:
            print("No A/B tests found.")
        print(f"{'='*70}")

    elif action == "status":
        if not test_id:
            # Check for active test
            active = manager.get_active_test()
            if active:
                test_id = active.test_id
            else:
                print("No active test. Specify --test-id")
                return

        result = manager.calculate_results(test_id)
        config = manager.get_test(test_id)

        print(f"\n{'='*60}")
        print(f"A/B Test Status: {test_id}")
        print(f"{'='*60}")
        print(f"Status: {config.status}")
        print(f"\n{'Metric':<25} {'Version A':<15} {'Version B'}")
        print("-" * 55)
        ma = result.version_a_metrics
        mb = result.version_b_metrics
        print(f"{'Version':<25} {ma.version:<15} {mb.version}")
        print(f"{'Requests':<25} {ma.requests:<15} {mb.requests}")
        print(f"{'Positive Feedback':<25} {ma.positive_feedback:<15} {mb.positive_feedback}")
        print(f"{'Feedback Rate':<25} {ma.feedback_rate*100:.1f}%{'':<9} {mb.feedback_rate*100:.1f}%")
        print(f"{'Avg Latency (ms)':<25} {ma.avg_latency_ms:.1f}{'':<9} {mb.avg_latency_ms:.1f}")
        print(f"\nStatistical significance: {result.statistical_significance*100:.1f}%")
        print(f"Recommendation: {result.recommendation}")
        print(f"{'='*60}")

    elif action == "complete":
        if not test_id:
            print("Error: --test-id required")
            return

        manager.complete_test(test_id, winner)
        print(f"Test {test_id} completed.")
        if winner:
            print(f"Winner: {winner}")

    elif action == "cancel":
        if not test_id:
            print("Error: --test-id required")
            return

        manager.cancel_test(test_id)
        print(f"Test {test_id} cancelled.")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: create, list, status, complete, cancel")


def validate_command(version: str = None):
    """Model validation command."""
    from .validation import get_validator, save_default_prompts
    from .version_manager import get_version_manager

    validator = get_validator()
    vm = get_version_manager()

    if version is None:
        version = vm.get_active_version()
        if not version:
            print("No version specified and no active version found.")
            return

    adapter_path = vm.get_version_path(version)
    if not adapter_path:
        print(f"Version not found: {version}")
        return

    adapter_path = adapter_path / "adapter"

    print(f"\n{'='*50}")
    print(f"Validating Model: {version}")
    print(f"{'='*50}")

    # File-based validation
    result = validator.validate_files_only(adapter_path, version)

    print(f"\nFile Validation:")
    print(f"  Passed: {result.passed_tests}/{result.total_tests}")

    if result.failed_tests:
        print(f"\n  Failed checks:")
        for test in result.failed_tests:
            print(f"    - {test.failure_reason}")

    print(f"\nOverall: {'PASSED' if result.passed else 'FAILED'}")
    print(f"{'='*50}")

    # Save result
    validator.save_validation_result(result)


def correction_command(action: str, correction_id: str = None, output: str = None):
    """Response correction commands."""
    from .corrections import get_correction_manager

    manager = get_correction_manager()

    if action == "list":
        corrections = manager.get_corrections(limit=20)
        print(f"\n{'='*70}")
        print("Recent Corrections")
        print(f"{'='*70}")
        print(f"{'ID':<30} {'Type':<15} {'Date'}")
        print("-" * 70)
        for corr in corrections:
            print(f"{corr.id:<30} {corr.correction_type:<15} {corr.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"\nTotal: {len(corrections)}")
        print(f"{'='*70}")

    elif action == "stats":
        stats = manager.get_statistics()
        print(f"\n{'='*40}")
        print("Correction Statistics")
        print(f"{'='*40}")
        print(f"Total corrections: {stats['total']}")
        print(f"This month: {stats['this_month']}")
        print(f"\nBy type:")
        for ctype, count in stats['by_type'].items():
            print(f"  {ctype}: {count}")
        print(f"{'='*40}")

    elif action == "export":
        output_path = Path(output) if output else None
        path = manager.export_for_training(output_path)
        print(f"Exported corrections to: {path}")

    elif action == "show":
        if not correction_id:
            print("Error: --id required")
            return

        corr = manager.get_correction(correction_id)
        if not corr:
            print(f"Correction not found: {correction_id}")
            return

        print(f"\n{'='*60}")
        print(f"Correction: {corr.id}")
        print(f"{'='*60}")
        print(f"Type: {corr.correction_type}")
        print(f"Session: {corr.session_id}")
        print(f"Date: {corr.timestamp}")
        print(f"\nOriginal prompt:\n  {corr.original_prompt}")
        print(f"\nOriginal response:\n  {corr.original_response[:200]}...")
        print(f"\nCorrected response:\n  {corr.corrected_response[:200]}...")
        if corr.notes:
            print(f"\nNotes: {corr.notes}")
        print(f"{'='*60}")

    elif action == "delete":
        if not correction_id:
            print("Error: --id required")
            return

        if manager.delete_correction(correction_id):
            print(f"Deleted correction: {correction_id}")
        else:
            print(f"Correction not found: {correction_id}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: list, stats, export, show, delete")


def main():
    parser = argparse.ArgumentParser(description="JAVIS Model Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List model versions")
    list_parser.add_argument("--dir", default="models", help="Models directory")

    # export command
    export_parser = subparsers.add_parser("export", help="Export training data")
    export_parser.add_argument("--output", "-o", help="Output file path")
    export_parser.add_argument("--filter", choices=["good", "bad"], help="Filter by feedback")

    # version command
    version_parser = subparsers.add_parser("version", help="Create new version")
    version_parser.add_argument("adapter_path", help="Path to adapter")
    version_parser.add_argument("--name", help="Version name")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")

    # scheduler command
    scheduler_parser = subparsers.add_parser("scheduler", help="Control training scheduler")
    scheduler_parser.add_argument(
        "action",
        choices=["start", "stop", "status", "trigger"],
        help="Scheduler action"
    )

    # train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--provider",
        choices=["modal", "local"],
        help="Training provider (default: from config)"
    )
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip condition checks"
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without training"
    )

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version")
    rollback_parser.add_argument(
        "--version",
        help="Specific version to rollback to"
    )

    # active command
    active_parser = subparsers.add_parser("active", help="Show active model version")

    # Phase 5: quality command
    quality_parser = subparsers.add_parser("quality", help="Data quality management")
    quality_parser.add_argument(
        "action",
        choices=["stats", "score-all", "export", "high-quality"],
        help="Quality action"
    )
    quality_parser.add_argument(
        "--min-score",
        type=float,
        default=3.5,
        help="Minimum score threshold"
    )
    quality_parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )

    # Phase 5: metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Performance metrics")
    metrics_parser.add_argument(
        "action",
        choices=["show", "compare", "dashboard", "cleanup"],
        help="Metrics action"
    )
    metrics_parser.add_argument("--version", help="Model version")
    metrics_parser.add_argument("--version-a", help="First version for comparison")
    metrics_parser.add_argument("--version-b", help="Second version for comparison")
    metrics_parser.add_argument("--days", type=int, default=7, help="Number of days")

    # Phase 5: ab-test command
    abtest_parser = subparsers.add_parser("ab-test", help="A/B testing")
    abtest_parser.add_argument(
        "action",
        choices=["create", "list", "status", "complete", "cancel"],
        help="A/B test action"
    )
    abtest_parser.add_argument("--version-a", help="Control version")
    abtest_parser.add_argument("--version-b", help="Treatment version")
    abtest_parser.add_argument("--test-id", help="Test ID")
    abtest_parser.add_argument("--split", type=float, default=0.5, help="Traffic split for version B")
    abtest_parser.add_argument("--winner", help="Declare winner when completing")

    # Phase 5: validate command
    validate_parser = subparsers.add_parser("validate", help="Validate model")
    validate_parser.add_argument("version", nargs="?", help="Version to validate")

    # Phase 5: correction command
    correction_parser = subparsers.add_parser("correction", help="Response corrections")
    correction_parser.add_argument(
        "action",
        choices=["list", "stats", "export", "show", "delete"],
        help="Correction action"
    )
    correction_parser.add_argument("--id", dest="correction_id", help="Correction ID")
    correction_parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    if args.command == "list":
        list_versions(args.dir)
    elif args.command == "export":
        export_data(args.output, args.filter)
    elif args.command == "version":
        create_version(args.adapter_path, args.name)
    elif args.command == "stats":
        stats()
    elif args.command == "scheduler":
        scheduler_command(args.action)
    elif args.command == "train":
        train_command(
            provider=args.provider,
            force=args.force,
            dry_run=args.dry_run
        )
    elif args.command == "rollback":
        rollback_command(args.version)
    elif args.command == "active":
        active_command()
    # Phase 5 commands
    elif args.command == "quality":
        quality_command(
            action=args.action,
            min_score=args.min_score,
            output=args.output
        )
    elif args.command == "metrics":
        metrics_command(
            action=args.action,
            version=args.version,
            version_a=getattr(args, 'version_a', None),
            version_b=getattr(args, 'version_b', None),
            days=args.days
        )
    elif args.command == "ab-test":
        abtest_command(
            action=args.action,
            version_a=getattr(args, 'version_a', None),
            version_b=getattr(args, 'version_b', None),
            test_id=getattr(args, 'test_id', None),
            split=args.split,
            winner=getattr(args, 'winner', None)
        )
    elif args.command == "validate":
        validate_command(version=args.version)
    elif args.command == "correction":
        correction_command(
            action=args.action,
            correction_id=args.correction_id,
            output=args.output
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
