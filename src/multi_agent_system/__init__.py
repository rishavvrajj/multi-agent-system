from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .pipeline import run_research_pipeline

console = Console()


def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Multi-Agent Research Pipeline[/bold cyan]\n"
            "[dim]Autonomous Search, Scraping, Writing & Critical Review[/dim]",
            border_style="cyan",
        )
    )
    topic = Prompt.ask("\n[bold green]Enter a research topic[/bold green]")
    if topic.strip():
        run_research_pipeline(topic=topic.strip())


if __name__ == "__main__":
    main()