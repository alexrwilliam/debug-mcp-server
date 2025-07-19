"""Debug MCP Server - Main server implementation."""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiofiles
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# === Data Models ===

class ErrorDetail(BaseModel):
    """Detailed error information."""
    type: str
    message: str
    suggestions: List[str] = Field(default_factory=list)

class FileResult(BaseModel):
    """File operation result."""
    success: bool
    content: Optional[str] = None
    error: Optional[Union[str, ErrorDetail]] = None
    lines_read: Optional[int] = None
    file_info: Optional[Dict[str, Any]] = None

class CommandResult(BaseModel):
    """Command execution result."""
    success: bool
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: Optional[int] = None
    error: Optional[str] = None

class DebugSession(BaseModel):
    """Debug session information."""
    session_id: str
    project_path: str
    debug_directory: str
    created_at: str
    status: str

class SelectorAnalysis(BaseModel):
    """Code selector analysis result."""
    selectors: List[str] = Field(default_factory=list)
    selector_types: Dict[str, List[str]] = Field(default_factory=dict)
    analysis: Optional[str] = None

# === Global State ===
active_sessions: Dict[str, DebugSession] = {}
current_working_directory: Optional[Path] = None

# === MCP Server Setup ===
mcp = FastMCP("debug-mcp")

# === File Operations Tools ===

@mcp.tool()
async def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> FileResult:
    """Read file contents, optionally specifying line range."""
    try:
        file_path = Path(path).expanduser().resolve()
        
        if not file_path.exists():
            return FileResult(success=False, error=f"File not found: {path}")
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            if start_line is None and end_line is None:
                content = await f.read()
                lines_read = len(content.splitlines())
            else:
                lines = await f.readlines()
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else len(lines)
                content = ''.join(lines[start_idx:end_idx])
                lines_read = end_idx - start_idx
        
        return FileResult(
            success=True,
            content=content,
            lines_read=lines_read
        )
    
    except Exception as e:
        return FileResult(success=False, error=str(e))

@mcp.tool()
async def write_file(path: str, content: str) -> FileResult:
    """Write content to file, creating directories if needed."""
    try:
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        return FileResult(success=True)
    
    except Exception as e:
        return FileResult(success=False, error=str(e))

@mcp.tool()
async def search_file(path: str, pattern: str) -> Dict[str, Any]:
    """Search for pattern in file, return matching lines with line numbers."""
    try:
        file_path = Path(path).expanduser().resolve()
        
        if not file_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        
        matches = []
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            lines = await f.readlines()
            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    matches.append({
                        "line_number": line_num,
                        "line": line.strip(),
                        "match": re.search(pattern, line).group()
                    })
        
        return {
            "success": True,
            "matches": matches,
            "total_matches": len(matches)
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def edit_file(path: str, old_content: str, new_content: str) -> FileResult:
    """Edit file by replacing old_content with new_content."""
    try:
        file_path = Path(path).expanduser().resolve()
        
        if not file_path.exists():
            return FileResult(success=False, error=f"File not found: {path}")
        
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            current_content = await f.read()
        
        if old_content not in current_content:
            return FileResult(success=False, error="Old content not found in file")
        
        updated_content = current_content.replace(old_content, new_content)
        
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(updated_content)
        
        return FileResult(success=True)
    
    except Exception as e:
        return FileResult(success=False, error=str(e))

# === Enhanced File Operations ===

@mcp.tool()
async def list_directory(path: str = ".", pattern: Optional[str] = None) -> Dict[str, Any]:
    """List files and directories with detailed information."""
    try:
        dir_path = resolve_path(path)
        
        if not dir_path.exists():
            return {
                "success": False,
                "error": ErrorDetail(
                    type="DirectoryNotFoundError",
                    message=f"Directory not found: {path}",
                    suggestions=["Check directory path", "Use get_working_directory to see current location"]
                )
            }
        
        if not dir_path.is_dir():
            return {
                "success": False,
                "error": ErrorDetail(
                    type="NotADirectoryError",
                    message=f"Path is not a directory: {path}",
                    suggestions=["Use a directory path", "Check if path points to a file"]
                )
            }
        
        items = []
        for item in dir_path.iterdir():
            try:
                stat = item.stat()
                item_info = {
                    "name": item.name,
                    "path": str(item.relative_to(dir_path.parent)),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "permissions": oct(stat.st_mode)[-3:],
                    "hidden": item.name.startswith(".")
                }
                
                # Apply pattern filter if specified
                if pattern is None or re.search(pattern, item.name, re.IGNORECASE):
                    items.append(item_info)
                    
            except PermissionError:
                # Skip items we can't access
                continue
        
        # Sort by type then name
        items.sort(key=lambda x: (x["type"], x["name"].lower()))
        
        return {
            "success": True,
            "directory": str(dir_path),
            "items": items,
            "total_items": len(items),
            "pattern_filter": pattern
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": ErrorDetail(
                type=type(e).__name__,
                message=str(e),
                suggestions=["Check directory permissions", "Verify path format"]
            )
        }

@mcp.tool()
async def copy_file(src: str, dst: str, preserve_metadata: bool = True) -> FileResult:
    """Copy file from source to destination."""
    try:
        src_path = resolve_path(src)
        dst_path = resolve_path(dst)
        
        if not src_path.exists():
            return FileResult(
                success=False,
                error=ErrorDetail(
                    type="FileNotFoundError",
                    message=f"Source file not found: {src}",
                    suggestions=["Check source file path", "Verify file exists"]
                )
            )
        
        if not src_path.is_file():
            return FileResult(
                success=False,
                error=ErrorDetail(
                    type="NotAFileError",
                    message=f"Source is not a file: {src}",
                    suggestions=["Use file path, not directory", "Check source type"]
                )
            )
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        if preserve_metadata:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)
        
        # Get destination file info
        stat = dst_path.stat()
        file_info = {
            "source": str(src_path),
            "destination": str(dst_path),
            "size": stat.st_size,
            "copied_at": datetime.now().isoformat(),
            "metadata_preserved": preserve_metadata
        }
        
        return FileResult(
            success=True,
            file_info=file_info
        )
        
    except Exception as e:
        return FileResult(
            success=False,
            error=ErrorDetail(
                type=type(e).__name__,
                message=str(e),
                suggestions=["Check file permissions", "Verify destination directory", "Check disk space"]
            )
        )

# === Advanced Code Analysis ===

@mcp.tool()
async def analyze_code(code: str, language: str = "python") -> Dict[str, Any]:
    """Analyze code for syntax, structure, and dependencies."""
    try:
        analysis = {
            "language": language,
            "lines_of_code": len(code.splitlines()),
            "character_count": len(code),
            "syntax_valid": True,
            "syntax_errors": [],
            "imports": [],
            "functions": [],
            "classes": [],
            "dependencies": []
        }
        
        if language.lower() == "python":
            import ast
            try:
                tree = ast.parse(code)
                
                # Extract imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                            analysis["dependencies"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis["imports"].append(node.module)
                            analysis["dependencies"].append(node.module)
                    elif isinstance(node, ast.FunctionDef):
                        analysis["functions"].append({
                            "name": node.name,
                            "line": node.lineno,
                            "args": len(node.args.args),
                            "is_async": isinstance(node, ast.AsyncFunctionDef)
                        })
                    elif isinstance(node, ast.ClassDef):
                        analysis["classes"].append({
                            "name": node.name,
                            "line": node.lineno,
                            "methods": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
                        })
                        
            except SyntaxError as e:
                analysis["syntax_valid"] = False
                analysis["syntax_errors"].append({
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                    "type": "SyntaxError"
                })
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": ErrorDetail(
                type=type(e).__name__,
                message=str(e),
                suggestions=["Check code format", "Verify language parameter"]
            )
        }

@mcp.tool()
async def validate_syntax(code: str, language: str = "python") -> Dict[str, Any]:
    """Validate code syntax and provide error details."""
    try:
        if language.lower() == "python":
            import ast
            try:
                ast.parse(code)
                return {
                    "success": True,
                    "valid": True,
                    "language": language,
                    "message": "Code syntax is valid"
                }
            except SyntaxError as e:
                return {
                    "success": True,
                    "valid": False,
                    "language": language,
                    "error": {
                        "line": e.lineno,
                        "column": e.offset,
                        "message": e.msg,
                        "text": e.text.strip() if e.text else None
                    },
                    "suggestions": [
                        "Check syntax at specified line and column",
                        "Verify proper indentation",
                        "Check for missing colons, parentheses, or brackets"
                    ]
                }
        else:
            return {
                "success": False,
                "error": ErrorDetail(
                    type="UnsupportedLanguageError",
                    message=f"Syntax validation not supported for language: {language}",
                    suggestions=["Currently only Python syntax validation is supported"]
                )
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": ErrorDetail(
                type=type(e).__name__,
                message=str(e),
                suggestions=["Check code format", "Verify language parameter"]
            )
        }

# === Environment Management ===

@mcp.tool()
async def activate_venv(venv_path: str) -> Dict[str, Any]:
    """Activate virtual environment and return environment info."""
    try:
        venv_dir = resolve_path(venv_path)
        
        if not venv_dir.exists():
            return {
                "success": False,
                "error": ErrorDetail(
                    type="VirtualEnvNotFoundError",
                    message=f"Virtual environment not found: {venv_path}",
                    suggestions=["Check virtual environment path", "Create virtual environment first"]
                )
            }
        
        # Check for common venv structures
        activate_script = None
        python_executable = None
        
        if (venv_dir / "bin" / "activate").exists():  # Unix-like
            activate_script = venv_dir / "bin" / "activate"
            python_executable = venv_dir / "bin" / "python"
        elif (venv_dir / "Scripts" / "activate.bat").exists():  # Windows
            activate_script = venv_dir / "Scripts" / "activate.bat"
            python_executable = venv_dir / "Scripts" / "python.exe"
        
        if not activate_script:
            return {
                "success": False,
                "error": ErrorDetail(
                    type="InvalidVirtualEnvError",
                    message=f"No activation script found in: {venv_path}",
                    suggestions=["Check if this is a valid virtual environment", "Recreate virtual environment"]
                )
            }
        
        # Get Python version in venv
        python_version = "unknown"
        if python_executable and python_executable.exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    str(python_executable), "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                python_version = stdout.decode().strip()
            except:
                pass
        
        return {
            "success": True,
            "venv_path": str(venv_dir),
            "activate_script": str(activate_script),
            "python_executable": str(python_executable) if python_executable else None,
            "python_version": python_version,
            "instructions": f"To activate manually: source {activate_script}" if "bin" in str(activate_script) else f"To activate manually: {activate_script}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": ErrorDetail(
                type=type(e).__name__,
                message=str(e),
                suggestions=["Check virtual environment path", "Verify permissions"]
            )
        }

@mcp.tool()
async def install_package(package: str, venv_path: Optional[str] = None) -> CommandResult:
    """Install Python package using pip."""
    try:
        # Determine pip executable
        if venv_path:
            venv_dir = resolve_path(venv_path)
            pip_executable = venv_dir / "bin" / "pip"
            if not pip_executable.exists():
                pip_executable = venv_dir / "Scripts" / "pip.exe"
            if not pip_executable.exists():
                return CommandResult(
                    success=False,
                    error="pip not found in virtual environment"
                )
            pip_cmd = str(pip_executable)
        else:
            pip_cmd = "pip"
        
        # Install package
        command = f"{pip_cmd} install {package}"
        
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return CommandResult(
            success=process.returncode == 0,
            stdout=stdout.decode('utf-8') if stdout else None,
            stderr=stderr.decode('utf-8') if stderr else None,
            return_code=process.returncode
        )
        
    except Exception as e:
        return CommandResult(
            success=False,
            error=str(e)
        )

# === Terminal Operations Tools ===

@mcp.tool()
async def run_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: Optional[int] = 30
) -> CommandResult:
    """Execute shell command and return result."""
    try:
        work_dir = Path(cwd).expanduser().resolve() if cwd else None
        
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return CommandResult(
                success=False,
                error=f"Command timed out after {timeout} seconds"
            )
        
        return CommandResult(
            success=process.returncode == 0,
            stdout=stdout.decode('utf-8') if stdout else None,
            stderr=stderr.decode('utf-8') if stderr else None,
            return_code=process.returncode
        )
    
    except Exception as e:
        return CommandResult(success=False, error=str(e))

@mcp.tool()
async def run_python(
    script_path: str,
    args: Optional[List[str]] = None,
    venv: Optional[str] = None,
    timeout: Optional[int] = 60
) -> CommandResult:
    """Run Python script, optionally in virtual environment."""
    try:
        script_file = Path(script_path).expanduser().resolve()
        
        if not script_file.exists():
            return CommandResult(success=False, error=f"Script not found: {script_path}")
        
        # Build command
        if venv:
            venv_path = Path(venv).expanduser().resolve()
            if sys.platform == "win32":
                python_cmd = str(venv_path / "Scripts" / "python.exe")
            else:
                python_cmd = str(venv_path / "bin" / "python")
        else:
            python_cmd = sys.executable
        
        cmd_parts = [python_cmd, str(script_file)]
        if args:
            cmd_parts.extend(args)
        
        command = ' '.join(f'"{part}"' if ' ' in part else part for part in cmd_parts)
        
        return await run_command(command, cwd=str(script_file.parent), timeout=timeout)
    
    except Exception as e:
        return CommandResult(success=False, error=str(e))

# === Session Management Tools ===

@mcp.tool()
async def setup_debug_session(project_path: str) -> Dict[str, Any]:
    """Initialize debug session with artifact management."""
    try:
        session_id = str(uuid.uuid4())[:8]  # Shorter ID for readability
        project_dir = Path(project_path).expanduser().resolve()
        
        # Create debug artifacts directory
        debug_dir = project_dir / "debug_artifacts" / "sessions" / session_id
        
        # Create directory structure
        for subdir in ["responses", "code", "analysis", "logs"]:
            (debug_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create session metadata
        session = DebugSession(
            session_id=session_id,
            project_path=str(project_dir),
            debug_directory=str(debug_dir),
            created_at=str(asyncio.get_event_loop().time()),
            status="active"
        )
        
        active_sessions[session_id] = session
        
        # Write session metadata
        metadata_file = debug_dir / "metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(session.model_dump_json(indent=2))
        
        return {
            "success": True,
            "session_id": session_id,
            "debug_directory": str(debug_dir),
            "project_path": str(project_dir)
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

@mcp.tool()
async def get_session_artifacts(session_id: str) -> Dict[str, Any]:
    """Get artifacts and status for a debug session."""
    try:
        if session_id not in active_sessions:
            return {"success": False, "error": f"Session {session_id} not found"}
        
        session = active_sessions[session_id]
        debug_dir = Path(session.debug_directory)
        
        artifacts = {}
        for subdir in ["responses", "code", "analysis", "logs"]:
            subdir_path = debug_dir / subdir
            if subdir_path.exists():
                artifacts[subdir] = [
                    str(f.relative_to(debug_dir)) 
                    for f in subdir_path.iterdir() 
                    if f.is_file()
                ]
        
        return {
            "success": True,
            "session": session.model_dump(),
            "artifacts": artifacts
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

# === Code Analysis Tools ===

@mcp.tool()
async def extract_selectors(code: str) -> SelectorAnalysis:
    """Extract CSS and XPath selectors from Python code."""
    try:
        selectors = []
        selector_types = {"css": [], "xpath": [], "playwright": []}
        
        # CSS selectors (common patterns)
        css_patterns = [
            r'["\']([^"\']*[#.][^"\']*)["\']',  # Basic CSS selectors
            r'select\(["\']([^"\']+)["\']\)',    # .select() calls
            r'querySelector\(["\']([^"\']+)["\']\)',  # querySelector
        ]
        
        # XPath patterns
        xpath_patterns = [
            r'["\']//[^"\']*["\']',  # XPath expressions
            r'xpath\(["\']([^"\']+)["\']\)',  # xpath() calls
        ]
        
        # Playwright-specific patterns
        pw_patterns = [
            r'get_by_[a-z_]+\(["\']([^"\']+)["\']\)',  # get_by_* methods
            r'locator\(["\']([^"\']+)["\']\)',          # locator() calls
        ]
        
        for pattern in css_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if match and (match.startswith('#') or match.startswith('.') or ' ' in match):
                    selectors.append(match)
                    selector_types["css"].append(match)
        
        for pattern in xpath_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                selectors.append(match)
                selector_types["xpath"].append(match)
        
        for pattern in pw_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                selectors.append(match)
                selector_types["playwright"].append(match)
        
        analysis = f"Found {len(selectors)} selectors: {len(selector_types['css'])} CSS, {len(selector_types['xpath'])} XPath, {len(selector_types['playwright'])} Playwright"
        
        return SelectorAnalysis(
            selectors=list(set(selectors)),  # Remove duplicates
            selector_types=selector_types,
            analysis=analysis
        )
    
    except Exception as e:
        return SelectorAnalysis(analysis=f"Error analyzing selectors: {str(e)}")

# === Main Server Setup ===

def main():
    """Main server entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug MCP Server")
    parser.add_argument("transport", choices=["stdio", "http"], help="Transport type")
    parser.add_argument("--port", type=int, default=8932, help="HTTP port")
    parser.add_argument("--session-dir", help="Debug session directory")
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        mcp.run()
    else:
        # HTTP transport using StreamableHTTP
        import uvicorn
        app = mcp.streamable_http_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()