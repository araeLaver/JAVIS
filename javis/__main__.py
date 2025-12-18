"""
JAVIS Entry Point

Usage:
    python -m javis
    javis (after pip install)
"""

from javis.interfaces.cli import app


def main():
    """Main entry point for JAVIS CLI."""
    app()


if __name__ == "__main__":
    main()
