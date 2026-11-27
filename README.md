# Multi-Agent Research System

An autonomous multi-agent research pipeline that searches the web, scrapes deep sources, synthesizes structured reports, and critically evaluates the output.

## Architecture

The system executes a sequential 4-agent workflow:

```
[ User Input ]
       │
       ▼
1. Search Agent  ──► Tavily Web Search (discovers top sources & URLs)
       │
       ▼
2. Reader Agent  ──► Web Scraper (extracts deep content from best URL)
       │
       ▼
3. Writer Agent  ──► Report Synthesizer (drafts structured findings & sources)
       │
       ▼
4. Critic Agent  ──► Quality Reviewer (scores, critiques & evaluates report)
```

## Features

- **Interactive Streamlit Web UI:** Modern browser dashboard with live agent progress tracking, KPI metrics, tabbed findings, and one-click Markdown downloads.
- **Multi-Model Fallback:** Automatically cascades through Gemini models (`gemini-3.5-flash-lite` → `gemini-3.8-flash` → `gemini-3.1-flash-lite` → `gemini-3.7-flash`) to prevent 429 quota exhaustion and 503 service downtime.
- **Rich Terminal UI:** Formatted progress spinners, status banners, and rendered Markdown output panels for CLI users.
- **Clean Output Filtering:** Automatically strips metadata, cryptographic signatures, and token-heavy wrappers.

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Configure API Keys:**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

   *(Optional)* Customize model priority order:
   ```env
   GEMINI_MODELS=gemini-3.5-flash-lite,gemini-3.8-flash,gemini-3.1-flash-lite
   ```

## Usage

### 1. Launch the Streamlit Web Application (Recommended)

Start the interactive web dashboard in your browser:

```bash
uv run streamlit run app.py
```

*Or via the registered entry point:*
```bash
uv run multi-agent-system-ui
```

### 2. Run the Interactive Terminal CLI

```bash
uv run multi-agent-system
```
