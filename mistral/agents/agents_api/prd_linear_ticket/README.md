# PRD Generator & Linear Ticket Creator with MistralAI Agents API and MCP

Application that generates Product Requirements Documents (PDF) from transcript PDFs and creates Linear tickets using MistralAI Agents API with MCP servers.

[![Linear Ticket](https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/gif/Linear_tickets.gif)](https://www.youtube.com/watch?v=4UPP-JEjcKo)

## Use Case

- Extract text from PDF transcripts using OCR
- Generate comprehensive PRDs from meeting transcripts
- Parse PRDs into structured features
- Create Linear tickets automatically for each feature

## Architecture

```
├── mcp_servers/
│   ├── stdio_prd_generator_server.py
│   └── stdio_linear_ticket_gen_server.py
├── app.py
```

### Main Application
- **app.py**: Chainlit interface with MistralAI agent integration

### MCP Servers
- **stdio_prd_generator_server.py**: PDF OCR processing and PRD generation using MistralAI LLMs
- **stdio_linear_ticket_gen_server.py**: PRD parsing and Linear ticket creation via GraphQL API

## Installation

```bash
pip install chainlit mcp loguru pydantic gql
```

## Environment Setup

Set your MistralAI API key:
```bash
export MISTRAL_API_KEY="your_api_key_here"
```

Set your Mistral API key in both the servers:
- `mcp_servers/stdio_linear_ticket_gen_server.py`
- `mcp_servers/stdio_prd_generator_server`

Configure Linear API credentials in `mcp_servers/stdio_linear_ticket_gen_server.py`:
- Update `linear_api_key` with your Linear API key
- Update `team_id` with your Linear team ID

## Usage

Run the application:
```bash
chainlit run app.py
```

Ask questions like:
- "Generate a PRD from transcript.pdf"
- "Generate a PRD outlining LeChat improvements based on the transcript in transcript.pdf, and create corresponding Linear tickets."

The app will automatically extract text from PDFs, generate PRDs, and create corresponding Linear tickets.
