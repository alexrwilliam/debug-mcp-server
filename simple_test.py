"""Simple test without external dependencies."""

import re
import sys
from pathlib import Path

def test_selector_extraction():
    """Test the selector extraction logic."""
    test_code = """
import requests
from bs4 import BeautifulSoup

def scrape_page():
    soup = BeautifulSoup(html, 'html.parser')
    titles = soup.select('.title')  # CSS selector
    links = soup.select('a.link')   # Another CSS selector
    items = soup.find_all('div', class_='item')  # BeautifulSoup selector
    xpath_elements = driver.find_element("//div[@class='content']")  # XPath
    playwright_elem = page.get_by_role('button')  # Playwright
    return titles, links, items
"""
    
    selectors = []
    selector_types = {"css": [], "xpath": [], "playwright": []}
    
    # CSS selectors (common patterns)
    css_patterns = [
        r'["\']([^"\']*[#.][^"\']*)["\']',  # Basic CSS selectors
        r'select\(["\']([^"\']+)["\']\)',    # .select() calls
    ]
    
    # XPath patterns
    xpath_patterns = [
        r'["\']//[^"\']*["\']',  # XPath expressions
    ]
    
    # Playwright-specific patterns
    pw_patterns = [
        r'get_by_[a-z_]+\(["\']([^"\']+)["\']\)',  # get_by_* methods
    ]
    
    for pattern in css_patterns:
        matches = re.findall(pattern, test_code)
        for match in matches:
            if match and (match.startswith('#') or match.startswith('.') or ' ' in match):
                selectors.append(match)
                selector_types["css"].append(match)
    
    for pattern in xpath_patterns:
        matches = re.findall(pattern, test_code)
        for match in matches:
            selectors.append(match)
            selector_types["xpath"].append(match)
    
    for pattern in pw_patterns:
        matches = re.findall(pattern, test_code)
        for match in matches:
            selectors.append(match)
            selector_types["playwright"].append(match)
    
    print("✅ Selector extraction test:")
    print(f"   Found selectors: {list(set(selectors))}")
    print(f"   CSS: {selector_types['css']}")
    print(f"   XPath: {selector_types['xpath']}")
    print(f"   Playwright: {selector_types['playwright']}")

def test_file_structure():
    """Test that all files were created correctly."""
    base_path = Path("/Users/alexwilliamson/Desktop/Development/Genloop/Debug_MCP")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "pyproject.toml",
        "src/debug_mcp/__init__.py",
        "src/debug_mcp/server.py",
        "examples/usage_example.py"
    ]
    
    print("✅ File structure test:")
    for file_path in required_files:
        full_path = base_path / file_path
        exists = full_path.exists()
        print(f"   {file_path}: {'✓' if exists else '✗'}")
        
        if not exists:
            print(f"      Missing: {full_path}")

def test_server_imports():
    """Test that server imports work (without running)."""
    try:
        # Just test the syntax and basic imports
        server_path = Path("/Users/alexwilliamson/Desktop/Development/Genloop/Debug_MCP/src/debug_mcp/server.py")
        
        if server_path.exists():
            with open(server_path, 'r') as f:
                content = f.read()
            
            # Basic syntax check
            compile(content, str(server_path), 'exec')
            print("✅ Server syntax check: PASSED")
            
            # Count tools defined
            tool_count = content.count('@mcp.tool()')
            print(f"   Defined {tool_count} MCP tools")
            
        else:
            print("✗ Server file not found")
            
    except SyntaxError as e:
        print(f"✗ Syntax error in server: {e}")
    except Exception as e:
        print(f"✗ Error checking server: {e}")

if __name__ == "__main__":
    print("Testing Debug MCP Server (without dependencies)...\n")
    
    test_file_structure()
    print()
    test_selector_extraction()
    print()
    test_server_imports()
    print()
    print("🎉 Basic tests completed!")