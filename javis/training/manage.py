"""Model version management CLI."""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


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

    args = parser.parse_args()

    if args.command == "list":
        list_versions(args.dir)
    elif args.command == "export":
        export_data(args.output, args.filter)
    elif args.command == "version":
        create_version(args.adapter_path, args.name)
    elif args.command == "stats":
        stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
