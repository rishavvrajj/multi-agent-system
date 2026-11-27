import os
import warnings
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from .tools import scrape_url, web_search

load_dotenv()

# Suppress fixed sampling defaults warning from Gemini models
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")

# Configure primary and fallback models
DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

env_models = os.getenv("GEMINI_MODELS")
if env_models:
    models_to_use = [m.strip() for m in env_models.split(",") if m.strip()]
else:
    models_to_use = DEFAULT_MODELS


def create_llm_with_fallbacks(model_names: list[str]):
    """Build an LLM with graceful fallbacks across multiple Gemini models."""
    primary = ChatGoogleGenerativeAI(model=model_names[0])
    if len(model_names) > 1:
        fallbacks = [ChatGoogleGenerativeAI(model=m) for m in model_names[1:]]
        return primary.with_fallbacks(fallbacks)
    return primary


llm = create_llm_with_fallbacks(models_to_use)


def extract_text(content) -> str:
    """Extract clean human-readable text from LangChain message content, stripping signatures/extras."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a skilled web research agent. When given a topic, you MUST ALWAYS use the "
            "`web_search` tool to gather recent, reliable information and discover authoritative sources with URLs. "
            "Synthesize the findings clearly with URLs included."
        ),
    )


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a reading and scraping agent. Review the provided search results, select the most relevant "
            "and authoritative URL, and ALWAYS invoke the `scrape_url` tool on that URL to extract deep content. "
            "Summarize the extracted content concisely."
        ),
    )


writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful responses."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report: {report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()