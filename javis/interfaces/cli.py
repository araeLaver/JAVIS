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
    if not config.groq_api_key:
        print_error(
            "Groq API key not configured.\n"
            "Please set GROQ_API_KEY in .env file.\n"
            "Get a free key at: https://console.groq.com/keys"
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
def voice(
    tts_voice: str = typer.Option(
        None,
        "--tts-voice",
        "-v",
        help="TTS voice to use (e.g., ko-KR-InJoonNeural for male)",
    ),
):
    """Start voice conversation mode with JAVIS.

    Speak to JAVIS and hear responses via TTS.
    Say '종료' or press Ctrl+C to exit.
    """
    import asyncio

    # Load configuration
    load_config()
    config = get_config()

    # Check if voice is enabled
    if not config.voice.enabled:
        print_error(
            "Voice interface is not enabled.\n"
            "Set 'voice.enabled: true' in configs/config.yaml"
        )
        raise typer.Exit(1)

    # Check for API key (needed for STT)
    if not config.groq_api_key:
        print_error(
            "GROQ_API_KEY is required for voice mode.\n"
            "Please set GROQ_API_KEY in .env file."
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold blue]JAVIS[/bold blue] - Voice Mode\n"
            "[dim]음성으로 대화합니다.[/dim]\n"
            "  말하기 - 마이크에 대고 말하세요\n"
            "  종료   - '종료'라고 말하거나 Ctrl+C",
            title="🎤 Voice Mode",
            border_style="blue",
        )
    )

    try:
        from javis.voice.session import run_voice_mode

        asyncio.run(run_voice_mode(tts_voice))
    except ImportError as e:
        print_error(
            f"Voice dependencies not installed: {e}\n"
            "Install with: pip install javis[voice]"
        )
        raise typer.Exit(1)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(1)


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
        f"[bold]Groq API Key:[/bold] {'[green]Configured[/green]' if cfg.groq_api_key else '[red]Not set[/red]'}",
        title="Configuration",
    ))


# Workflow subcommands
workflow_app = typer.Typer(
    name="workflow",
    help="Manage and run workflows",
)
app.add_typer(workflow_app, name="workflow")


@workflow_app.command("list")
def workflow_list():
    """List all available workflows."""
    from rich.table import Table
    from javis.workflows import get_workflow_scheduler

    load_config()

    scheduler = get_workflow_scheduler()
    scheduler.load_workflows()

    workflows = scheduler.list_workflows()

    if not workflows:
        console.print("[dim]No workflows found.[/dim]")
        return

    table = Table(title="Workflows")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Enabled", justify="center")
    table.add_column("Trigger")
    table.add_column("Schedule")
    table.add_column("Last Run")
    table.add_column("Status")

    for wf in workflows:
        enabled = "[green]Yes[/green]" if wf.enabled else "[red]No[/red]"
        last_run = wf.last_run.strftime("%Y-%m-%d %H:%M") if wf.last_run else "-"
        status = wf.last_status or "-"
        status_color = "green" if status == "completed" else "red" if status == "failed" else "dim"

        table.add_row(
            wf.name,
            wf.description or "",
            enabled,
            wf.trigger_type,
            wf.trigger_schedule or "-",
            last_run,
            f"[{status_color}]{status}[/{status_color}]",
        )

    console.print(table)


@workflow_app.command("run")
def workflow_run(name: str = typer.Argument(..., help="Workflow name to run")):
    """Run a workflow immediately."""
    from javis.workflows import get_workflow_scheduler

    load_config()

    scheduler = get_workflow_scheduler()
    scheduler.load_workflows()

    workflow = scheduler.get_workflow(name)
    if not workflow:
        print_error(f"Workflow not found: {name}")
        raise typer.Exit(1)

    console.print(f"[bold]Running workflow:[/bold] {name}")

    with console.status("[bold green]Executing...[/bold green]"):
        result = scheduler.trigger_now(name)

    # Display result
    status_color = "green" if result.status.value == "completed" else "red"
    console.print(f"\n[bold]Status:[/bold] [{status_color}]{result.status.value}[/{status_color}]")
    console.print(f"[bold]Duration:[/bold] {result.duration:.2f}s")

    if result.steps:
        console.print("\n[bold]Steps:[/bold]")
        for step in result.steps:
            step_status = "[green]OK[/green]" if step.status.value == "success" else "[red]FAIL[/red]"
            console.print(f"  [{step_status}] {step.step_name}")
            if step.error:
                console.print(f"      [red]{step.error}[/red]")

    if result.error:
        console.print(f"\n[red]Error:[/red] {result.error}")


@workflow_app.command("status")
def workflow_status(name: str = typer.Argument(..., help="Workflow name")):
    """Show workflow status and details."""
    from javis.workflows import get_workflow_scheduler

    load_config()

    scheduler = get_workflow_scheduler()
    scheduler.load_workflows()

    workflow = scheduler.get_workflow(name)
    if not workflow:
        print_error(f"Workflow not found: {name}")
        raise typer.Exit(1)

    info = next((w for w in scheduler.list_workflows() if w.name == name), None)

    console.print(Panel(
        f"[bold]Name:[/bold] {workflow.name}\n"
        f"[bold]Description:[/bold] {workflow.description}\n"
        f"[bold]Enabled:[/bold] {'[green]Yes[/green]' if workflow.enabled else '[red]No[/red]'}\n"
        f"[bold]Trigger:[/bold] {workflow.trigger.type.value}\n"
        f"[bold]Schedule:[/bold] {workflow.trigger.cron or workflow.trigger.interval_seconds or 'Manual'}\n"
        f"[bold]Steps:[/bold] {len(workflow.steps)}\n"
        f"[bold]Tags:[/bold] {', '.join(workflow.tags) if workflow.tags else '-'}",
        title=f"Workflow: {name}",
    ))

    if info and info.last_run:
        console.print(f"\n[bold]Last Run:[/bold] {info.last_run.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"[bold]Last Status:[/bold] {info.last_status}")


@workflow_app.command("enable")
def workflow_enable(name: str = typer.Argument(..., help="Workflow name")):
    """Enable a workflow."""
    from javis.workflows import get_workflow_scheduler

    load_config()

    scheduler = get_workflow_scheduler()
    scheduler.load_workflows()

    if scheduler.enable_workflow(name):
        console.print(f"[green]Workflow '{name}' enabled.[/green]")
    else:
        print_error(f"Workflow not found: {name}")
        raise typer.Exit(1)


@workflow_app.command("disable")
def workflow_disable(name: str = typer.Argument(..., help="Workflow name")):
    """Disable a workflow."""
    from javis.workflows import get_workflow_scheduler

    load_config()

    scheduler = get_workflow_scheduler()
    scheduler.load_workflows()

    if scheduler.disable_workflow(name):
        console.print(f"[yellow]Workflow '{name}' disabled.[/yellow]")
    else:
        print_error(f"Workflow not found: {name}")
        raise typer.Exit(1)


@workflow_app.command("history")
def workflow_history(
    name: str = typer.Argument(..., help="Workflow name"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of entries to show"),
):
    """Show workflow execution history."""
    from rich.table import Table
    from javis.workflows import get_workflow_scheduler

    load_config()

    scheduler = get_workflow_scheduler()
    scheduler.load_workflows()

    workflow = scheduler.get_workflow(name)
    if not workflow:
        print_error(f"Workflow not found: {name}")
        raise typer.Exit(1)

    history = scheduler.get_history(name, limit=limit)

    if not history:
        console.print(f"[dim]No execution history for workflow '{name}'.[/dim]")
        return

    table = Table(title=f"Execution History: {name}")
    table.add_column("Run ID", style="dim")
    table.add_column("Started At")
    table.add_column("Duration")
    table.add_column("Status")
    table.add_column("Steps")

    for result in history:
        status_color = "green" if result.status.value == "completed" else "red"
        steps_ok = sum(1 for s in result.steps if s.status.value == "success")
        steps_total = len(result.steps)

        table.add_row(
            result.run_id[:8],
            result.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            f"{result.duration:.2f}s",
            f"[{status_color}]{result.status.value}[/{status_color}]",
            f"{steps_ok}/{steps_total}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
