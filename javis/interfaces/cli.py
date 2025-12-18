"""Command Line Interface for JAVIS."""

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from javis.utils.config import load_config, get_config
from javis.models.client import ModelClient, Message

app = typer.Typer(
    name="javis",
    help="JAVIS - Personal AI Assistant",
    add_completion=False,
)
console = Console()


class ChatSession:
    """Manages a chat session with JAVIS."""

    def __init__(self):
        self.config = get_config()
        self.client: ModelClient | None = None
        self.messages: list[Message] = []
        self._init_system_prompt()

    def _init_system_prompt(self):
        """Initialize with system prompt."""
        system_prompt = self.config.conversation.system_prompt
        self.messages = [Message(role="system", content=system_prompt)]

    def _init_client(self):
        """Initialize model client (lazy loading)."""
        if self.client is None:
            self.client = ModelClient()

    def chat(self, user_input: str) -> str:
        """Send a message and get a response."""
        self._init_client()

        # Add user message
        self.messages.append(Message(role="user", content=user_input))

        # Get response
        response = self.client.chat_sync(self.messages)

        # Add assistant message
        self.messages.append(Message(role="assistant", content=response.content))

        # Trim history if needed
        max_history = self.config.conversation.max_history
        if len(self.messages) > max_history + 1:  # +1 for system prompt
            self.messages = [self.messages[0]] + self.messages[-(max_history):]

        return response.content

    def clear(self):
        """Clear conversation history."""
        self._init_system_prompt()


def print_welcome():
    """Print welcome message."""
    console.print(
        Panel(
            "[bold blue]JAVIS[/bold blue] - Personal AI Assistant\n"
            "[dim]Type your message or use commands:[/dim]\n"
            "  /clear - Clear conversation\n"
            "  /exit  - Exit JAVIS",
            title="Welcome",
            border_style="blue",
        )
    )


def print_response(content: str):
    """Print assistant response."""
    console.print()
    console.print(Panel(Markdown(content), title="[bold green]JAVIS[/bold green]", border_style="green"))
    console.print()


def print_error(message: str):
    """Print error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


@app.command()
def chat():
    """Start an interactive chat session with JAVIS."""
    # Load configuration
    load_config()
    config = get_config()

    # Check for API keys
    if not config.runpod_api_key or not config.runpod_endpoint_id:
        print_error(
            "RunPod credentials not configured.\n"
            "Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID in .env file."
        )
        console.print("\n[dim]Copy .env.example to .env and fill in your credentials.[/dim]")
        raise typer.Exit(1)

    print_welcome()

    session = ChatSession()

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.strip().lower() == "/exit":
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.strip().lower() == "/clear":
                session.clear()
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            if user_input.strip().startswith("/"):
                console.print(f"[dim]Unknown command: {user_input}[/dim]")
                continue

            # Get response
            with console.status("[bold green]Thinking...[/bold green]"):
                response = session.chat(user_input)

            print_response(response)

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            print_error(str(e))


@app.command()
def version():
    """Show JAVIS version."""
    from javis import __version__

    console.print(f"JAVIS version {__version__}")


@app.command()
def config():
    """Show current configuration."""
    load_config()
    cfg = get_config()

    console.print(Panel(
        f"[bold]App:[/bold] {cfg.app.name} v{cfg.app.version}\n"
        f"[bold]Model:[/bold] {cfg.model.base_model}\n"
        f"[bold]Provider:[/bold] {cfg.model.provider}\n"
        f"[bold]Endpoint:[/bold] {'[green]Configured[/green]' if cfg.runpod_endpoint_id else '[red]Not set[/red]'}\n"
        f"[bold]API Key:[/bold] {'[green]Configured[/green]' if cfg.runpod_api_key else '[red]Not set[/red]'}",
        title="Configuration",
    ))


if __name__ == "__main__":
    app()
