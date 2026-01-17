# Provider-Agnostic Coding Agent

A bare-bones coding assistant that works with any LLM provider via litellm.

## Setup

```bash
uv sync
```

## Configuration



Set environment variables in `.env`:

```bash
cp .env.example .env
```

You can find more info on supported models [here](https://docs.litellm.ai/docs/providers).

### OpenAI
```bash
MODEL=gpt-4o
API_KEY=your-openai-api-key
```

### Anthropic
```bash
MODEL=claude-3-5-sonnet-20241022
API_KEY=your-anthropic-api-key
```

### Custom API
```bash
MODEL=your-custom-model
API_KEY=your-custom-api-key
API_BASE=https://your-custom-endpoint.com/v1
```

## Usage

```bash
uv run python agent.py
```

The agent provides file operations: read, list, and edit files. Chat naturally to perform coding tasks.