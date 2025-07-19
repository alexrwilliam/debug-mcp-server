"""Example usage of Debug MCP Server tools."""

import asyncio
import json
from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

async def test_debug_mcp():
    """Test basic Debug MCP functionality."""
    print("Testing Debug MCP Server...")
    
    # Connect to Debug MCP server (assuming it's running on port 8932)
    try:
        async with sse_client("http://localhost:8932/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                
                print(f"Loaded {len(tools)} tools:")
                for tool in tools:
                    print(f"  - {tool.name}")
                
                # Test file operations
                print("\n=== Testing File Operations ===")
                
                # Create a test file
                test_content = """
# Test Python script
import requests
from bs4 import BeautifulSoup

def scrape_data():
    url = "https://example.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract data using CSS selectors
    titles = soup.select('.title')
    links = soup.select('a.link')
    
    return {"titles": titles, "links": links}
"""
                
                # Find write_file tool
                write_tool = next(t for t in tools if t.name == "write_file")
                write_result = await write_tool.ainvoke({
                    "path": "/tmp/test_scraper.py",
                    "content": test_content
                })
                print(f"Write result: {write_result}")
                
                # Test read_file
                read_tool = next(t for t in tools if t.name == "read_file")
                read_result = await read_tool.ainvoke({
                    "path": "/tmp/test_scraper.py"
                })
                print(f"Read result success: {read_result.get('success', False)}")
                
                # Test selector extraction
                extract_tool = next(t for t in tools if t.name == "extract_selectors")
                selector_result = await extract_tool.ainvoke({
                    "code": test_content
                })
                print(f"Extracted selectors: {selector_result}")
                
                # Test search
                search_tool = next(t for t in tools if t.name == "search_file")
                search_result = await search_tool.ainvoke({
                    "path": "/tmp/test_scraper.py",
                    "pattern": "select"
                })
                print(f"Search matches: {search_result.get('total_matches', 0)}")
                
                # Test command execution
                print("\n=== Testing Terminal Operations ===")
                cmd_tool = next(t for t in tools if t.name == "run_command")
                cmd_result = await cmd_tool.ainvoke({
                    "command": "python --version"
                })
                print(f"Python version: {cmd_result.get('stdout', '').strip()}")
                
                # Test session management
                print("\n=== Testing Session Management ===")
                session_tool = next(t for t in tools if t.name == "setup_debug_session")
                session_result = await session_tool.ainvoke({
                    "project_path": "/tmp/test_project"
                })
                print(f"Debug session created: {session_result}")
                
                print("\n✅ All tests completed successfully!")
                
    except Exception as e:
        print(f"❌ Error testing Debug MCP: {e}")
        print("Make sure the Debug MCP server is running on port 8932")

if __name__ == "__main__":
    asyncio.run(test_debug_mcp())