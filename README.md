# Debug MCP Server

A Model Context Protocol (MCP) server that provides debugging and development operations for AI-driven scraper development workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Features

- **📁 File Operations**: Read, write, search, and edit files with line-level precision
- **💻 Terminal Execution**: Run commands, Python scripts, and manage virtual environments  
- **🔍 Code Analysis**: Extract CSS/XPath selectors, validate syntax, and analyze code patterns
- **📊 Session Management**: Persistent debug sessions with structured artifact management
- **🔗 Integration Ready**: Designed to work seamlessly with Playwright MCP and HTTP MCP servers

## 📦 Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/alexwilliamson/debug-mcp-server.git
```

### For Development

```bash
git clone https://github.com/alexwilliamson/debug-mcp-server.git
cd debug-mcp-server
pip install -e .
```

## 🏃 Quick Start

### 1. Start the Server

```bash
# HTTP transport (recommended for AI agents)
debug-mcp http --port 8932

# Or stdio transport (for direct MCP clients)
debug-mcp stdio
```

### 2. Connect from AI Agent

```python
from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

async with sse_client("http://localhost:8932/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        debug_tools = await load_mcp_tools(session)
        
        # Now you have 9 powerful debugging tools available!
        for tool in debug_tools:
            print(f"Available: {tool.name}")
```

### 3. Use in Debug Workflows

This server powers automated scraper debugging workflows where AI agents:
1. Generate scraper code
2. Execute and test it
3. Analyze errors automatically
4. Generate and apply fixes
5. Retry until working

## 🛠️ Available Tools

### File Operations
| Tool | Description | Example Use |
|------|-------------|-------------|
| `read_file` | Read file contents with optional line ranges | Read scraper code, debug responses |
| `write_file` | Write content to files | Save generated scrapers, debug scripts |
| `search_file` | Search for regex patterns in files | Find CSS selectors, error patterns |
| `edit_file` | Replace content in existing files | Apply targeted code fixes |

### Terminal Operations
| Tool | Description | Example Use |
|------|-------------|-------------|
| `run_command` | Execute shell commands with timeout | Run scrapers, install packages |
| `run_python` | Run Python scripts in optional venv | Test scraper execution |

### Code Analysis
| Tool | Description | Example Use |
|------|-------------|-------------|
| `extract_selectors` | Find CSS/XPath/Playwright selectors in code | Audit scraper selectors |

### Session Management
| Tool | Description | Example Use |
|------|-------------|-------------|
| `setup_debug_session` | Initialize debug session with artifacts | Start debugging workflow |
| `get_session_artifacts` | List all session files and data | Review debug history |

## 🏗️ Architecture Integration

This server is part of a complete AI scraper debugging stack:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Playwright MCP │    │   Debug MCP     │    │   HTTP MCP      │
│  Browser Auto   │    │  This Server    │    │  Network Test   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │    AI Agent         │
                    │                     │
                    │ 1. Plan Website     │
                    │ 2. Generate Code    │
                    │ 3. Debug & Fix      │
                    │ 4. Deliver Scraper  │
                    └─────────────────────┘
```

## 🔧 Configuration

### Command Line Options

```bash
debug-mcp --help

# Common configurations:
debug-mcp http --port 8932                    # HTTP on custom port
debug-mcp stdio                               # Stdio for direct integration
debug-mcp http --session-dir ./my_sessions    # Custom session directory
```

### Session Artifacts

Debug sessions create structured artifact directories:

```
debug_artifacts/
└── sessions/
    └── {session_id}/
        ├── metadata.json       # Session info
        ├── responses/          # HTTP responses, HTML files
        ├── code/              # Generated scrapers, fixes
        ├── analysis/          # Error analysis, reports  
        └── logs/              # Execution logs
```

## 🤝 Integration Examples

### With LangGraph Agents

```python
from debug_subgraph import run_debug_workflow
from scraping_plan_models import PageAnalysis

# Your AI agent creates a scraping plan
plan = PageAnalysis(url="...", selectors=["..."], ...)

# Debug workflow automatically implements and tests
result = await run_debug_workflow(plan, goal, url)

if result["success"]:
    print(f"Working scraper: {result['scraper_path']}")
else:
    print(f"Debug failed: {result['error']}")
```

### Standalone Usage

```python
# Setup debug session
session = await setup_debug_session("/path/to/project")

# Generate and test code
scraper_code = "..."
await write_file(f"{session['debug_directory']}/scraper.py", scraper_code)
result = await run_python(f"{session['debug_directory']}/scraper.py")

# Analyze and fix if needed
if not result["success"]:
    errors = await search_file("debug.log", "ERROR")
    # Apply fixes based on error analysis
```

## 🧪 Testing

```bash
# Test the server directly
python -m debug_mcp.server http --port 8932

# In another terminal:
curl http://localhost:8932/health
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies installed with `pip install -r requirements.txt`
2. **Port Conflicts**: Use different port with `--port 8933`
3. **Permission Errors**: Check file system permissions for session directories
4. **Timeout Issues**: Increase timeout for long-running operations

### Debug Mode

Set logging level for detailed output:

```bash
export PYTHONPATH=/path/to/debug-mcp-server/src
debug-mcp http --port 8932 --log-level DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for AI-driven development workflows
- Integrates with [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)
- Designed for [LangGraph](https://github.com/langchain-ai/langgraph) agent workflows
- Part of automated scraper development pipeline