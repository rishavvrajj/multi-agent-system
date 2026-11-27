from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agents import (
    build_reader_agent,
    build_search_agent,
    critic_chain,
    extract_text,
    writer_chain,
)

console = Console()


def run_research_pipeline(topic: str) -> dict:
    state = {}

    console.print()
    console.rule("[bold cyan]Step 1: Search Agent Gathering Sources[/bold cyan]")
    with console.status("[cyan]Searching the web for reliable information...[/cyan]"):
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })

    state["search_results"] = extract_text(search_result["messages"][-1].content)
    console.print(
        Panel(
            state["search_results"],
            title="[bold cyan]Search Results & Sources[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print()
    console.rule("[bold blue]Step 2: Reader Agent Scraping Deep Content[/bold blue]")
    with console.status("[blue]Scraping and analyzing top resources...[/blue]"):
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results']}"
            )]
        })

    state["scraped_results"] = extract_text(reader_result["messages"][-1].content)
    console.print(
        Panel(
            state["scraped_results"],
            title="[bold blue]Scraped Resource Insights[/bold blue]",
            border_style="blue",
        )
    )

    console.print()
    console.rule("[bold green]Step 3: Writer Drafting Research Report[/bold green]")
    with console.status("[green]Synthesizing findings into a structured report...[/green]"):
        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_results']}"
        )
        report_output = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        state["report"] = extract_text(report_output)

    console.print(
        Panel(
            Markdown(state["report"]),
            title="[bold green]Final Research Report[/bold green]",
            border_style="green",
        )
    )

    console.print()
    console.rule("[bold yellow]Step 4: Critic Reviewing Report[/bold yellow]")
    with console.status("[yellow]Evaluating report accuracy and structure...[/yellow]"):
        feedback_output = critic_chain.invoke({
            "report": state["report"],
        })
        state["feedback"] = extract_text(feedback_output)

    console.print(
        Panel(
            Markdown(state["feedback"]),
            title="[bold yellow]Critic Evaluation[/bold yellow]",
            border_style="yellow",
        )
    )

    console.print()
    console.rule("[bold green]Pipeline Complete[/bold green]")
    return state