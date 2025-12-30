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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
