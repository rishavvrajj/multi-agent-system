from typing import Any, Generator
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agents import (
    build_reader_agent,
    build_search_agent,
    create_llm_with_fallbacks,
    extract_text,
    get_critic_chain,
    get_writer_chain,
)

console = Console()


def run_research_pipeline_stream(
    topic: str,
    model_names: list[str] | None = None,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Execute the 4-agent research pipeline yielding progress events at each stage."""
    custom_llm = create_llm_with_fallbacks(model_names) if model_names else None
    state: dict[str, Any] = {}

    # Step 1: Search Agent Gathering Sources
    yield {
        "step": "search",
        "status": "running",
        "message": "Searching the web for reliable information...",
    }
    search_agent = build_search_agent(custom_llm=custom_llm)
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = extract_text(search_result["messages"][-1].content)
    yield {
        "step": "search",
        "status": "complete",
        "message": "Discovered authoritative sources and search results",
        "data": state["search_results"],
    }

    # Step 2: Reader Agent Scraping Deep Content
    yield {
        "step": "reader",
        "status": "running",
        "message": "Scraping and analyzing top resources...",
    }
    reader_agent = build_reader_agent(custom_llm=custom_llm)
    reader_result = reader_agent.invoke({
        "messages": [(
            "user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results']}"
        )]
    })
    state["scraped_results"] = extract_text(reader_result["messages"][-1].content)
    yield {
        "step": "reader",
        "status": "complete",
        "message": "Extracted deep web content",
        "data": state["scraped_results"],
    }

    # Step 3: Writer Drafting Research Report
    yield {
        "step": "writer",
        "status": "running",
        "message": "Synthesizing findings into a structured report...",
    }
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_results']}"
    )
    writer_chain = get_writer_chain(custom_llm=custom_llm)
    report_output = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })
    state["report"] = extract_text(report_output)
    yield {
        "step": "writer",
        "status": "complete",
        "message": "Drafted structured research report",
        "data": state["report"],
    }

    # Step 4: Critic Reviewing Report
    yield {
        "step": "critic",
        "status": "running",
        "message": "Evaluating report accuracy, structure, and quality...",
    }
    critic_chain = get_critic_chain(custom_llm=custom_llm)
    feedback_output = critic_chain.invoke({
        "report": state["report"],
    })
    state["feedback"] = extract_text(feedback_output)
    yield {
        "step": "critic",
        "status": "complete",
        "message": "Critic evaluation complete",
        "data": state["feedback"],
    }

    yield {
        "step": "done",
        "status": "complete",
        "message": "Pipeline completed successfully",
        "state": state,
    }

    return state


def run_research_pipeline(topic: str) -> dict:
    """Run research pipeline synchronously with Rich terminal output (CLI mode)."""
    state = {}
    console.print()

    for event in run_research_pipeline_stream(topic):
        step = event.get("step")
        status = event.get("status")
        msg = event.get("message", "")
        data = event.get("data")

        if step == "search":
            if status == "running":
                console.rule("[bold cyan]Step 1: Search Agent Gathering Sources[/bold cyan]")
                console.print(f"[cyan]{msg}[/cyan]")
            elif status == "complete":
                state["search_results"] = data
                console.print(
                    Panel(
                        state["search_results"],
                        title="[bold cyan]Search Results & Sources[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print()

        elif step == "reader":
            if status == "running":
                console.rule("[bold blue]Step 2: Reader Agent Scraping Deep Content[/bold blue]")
                console.print(f"[blue]{msg}[/blue]")
            elif status == "complete":
                state["scraped_results"] = data
                console.print(
                    Panel(
                        state["scraped_results"],
                        title="[bold blue]Scraped Resource Insights[/bold blue]",
                        border_style="blue",
                    )
                )
                console.print()

        elif step == "writer":
            if status == "running":
                console.rule("[bold green]Step 3: Writer Drafting Research Report[/bold green]")
                console.print(f"[green]{msg}[/green]")
            elif status == "complete":
                state["report"] = data
                console.print(
                    Panel(
                        Markdown(state["report"]),
                        title="[bold green]Final Research Report[/bold green]",
                        border_style="green",
                    )
                )
                console.print()

        elif step == "critic":
            if status == "running":
                console.rule("[bold yellow]Step 4: Critic Reviewing Report[/bold yellow]")
                console.print(f"[yellow]{msg}[/yellow]")
            elif status == "complete":
                state["feedback"] = data
                console.print(
                    Panel(
                        Markdown(state["feedback"]),
                        title="[bold yellow]Critic Evaluation[/bold yellow]",
                        border_style="yellow",
                    )
                )
                console.print()

        elif step == "done":
            console.rule("[bold green]Pipeline Complete[/bold green]")
            return event.get("state", state)

    return state
