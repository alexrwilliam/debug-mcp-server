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

# === Helper Functions ===

def resolve_path(path: str) -> Path:
    """Resolve path relative to current working directory if set."""
    global current_working_directory
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.expanduser().resolve()
    else:
        base_dir = current_working_directory or Path.cwd()
        return (base_dir / path).resolve()

# === MCP Server Setup ===
mcp = FastMCP("debug-mcp")

# === File Operations Tools ===

@mcp.tool()
async def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> FileResult:
    """Read file contents with optional line range filtering.
    
    This tool reads text files from the filesystem with support for tilde expansion
    and relative paths. Can optionally return specific line ranges for examining
    large files or focusing on specific sections.
    
    Args:
        path: File path to read - supports ~ expansion and relative paths (e.g., "~/project/file.py", "./src/main.py")
        start_line: Optional starting line number (1-based, inclusive) to begin reading from
        end_line: Optional ending line number (1-based, inclusive) to stop reading at
    
    Returns:
        FileResult with success status, content string, lines_read count, and optional error message
    """
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
    """Write content to a file, automatically creating parent directories if needed.
    
    This tool writes text content to a file at the specified path. It will create
    any missing parent directories in the path automatically.
    
    Args:
        path: File path to write to - supports ~ expansion and relative paths (e.g., "~/project/new_file.py")
        content: Text content to write to the file
    
    Returns:
        FileResult with success status and optional error message
    """
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
    """Search for a regular expression pattern in a file and return matching lines.
    
    This tool searches through a file using regular expressions and returns all
    matching lines with their line numbers and the matched text.
    
    Args:
        path: File path to search in - supports ~ expansion and relative paths
        pattern: Regular expression pattern to search for (e.g., "class \\w+", "def test_.*", "TODO.*")
    
    Returns:
        Dict with success status, matches array (containing line_number, line, match), and total_matches count
    """
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
    """Edit a file by replacing specific text content with new content.
    
    This tool performs text replacement in files by finding exact matches of
    old_content and replacing with new_content. Useful for making targeted changes.
    
    Args:
        path: File path to edit - supports ~ expansion and relative paths
        old_content: Exact text content to find and replace (must match exactly including whitespace)
        new_content: New text content to replace the old content with
    
    Returns:
        FileResult with success status and optional error message if old_content not found
    """
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
    """List files and directories with detailed metadata and optional pattern filtering.
    
    This tool provides comprehensive directory listings including file sizes, permissions,
    modification times, and type classification. Supports regex pattern filtering.
    
    Args:
        path: Directory path to list (defaults to current directory ".") - supports ~ expansion and relative paths
        pattern: Optional regex pattern to filter items by name (case-insensitive, e.g., "*.py", "test.*")
    
    Returns:
        Dict with success status, directory path, items array (with name/path/type/size/modified/permissions/hidden), total_items count, and pattern_filter used
    """
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
    """Copy a file from source to destination with optional metadata preservation.
    
    This tool copies files between locations, automatically creating destination directories
    if needed. Can preserve or ignore file timestamps and permissions.
    
    Args:
        src: Source file path to copy from - supports ~ expansion and relative paths
        dst: Destination file path to copy to - supports ~ expansion and relative paths  
        preserve_metadata: If True, preserves timestamps and permissions; if False, only copies content (default: True)
    
    Returns:
        FileResult with success status, file_info containing source/destination/size/copied_at/metadata_preserved, and optional error
    """
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
    """Perform comprehensive static analysis of code including syntax, structure, and dependencies.
    
    This tool analyzes source code to extract metadata such as imports, functions, classes,
    syntax validation, and dependency information. Currently supports Python with extensible
    architecture for other languages.
    
    Args:
        code: Source code text to analyze (can be full file content or code snippets)
        language: Programming language of the code (currently supports "python", default: "python")
    
    Returns:
        Dict with success status and detailed analysis containing language, lines_of_code, character_count, syntax_valid, syntax_errors, imports, functions, classes, and dependencies arrays
    """
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
    """Validate source code syntax and provide detailed error information.
    
    This tool checks code for syntax errors and provides precise error locations
    and helpful suggestions for fixing issues. Essential for debugging syntax problems.
    
    Args:
        code: Source code text to validate (can be full file content or code snippets)
        language: Programming language to validate (currently supports "python", default: "python")
    
    Returns:
        Dict with success status, valid boolean, language, error details (line/column/message/text), and suggestions array for fixing issues
    """
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
    """Analyze and validate a Python virtual environment for activation.
    
    This tool examines a virtual environment directory to verify its structure,
    locate activation scripts, and provide environment information. Note: This analyzes
    the venv but doesn't actually activate it in the current process.
    
    Args:
        venv_path: Path to the virtual environment directory - supports ~ expansion and relative paths (e.g., "./venv", "~/project/env")
    
    Returns:
        Dict with success status, venv_path, activate_script location, python_executable path, python_version, and manual activation instructions
    """
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
    """Install a Python package using pip, optionally within a virtual environment.
    
    This tool installs Python packages using pip. If a virtual environment path is provided,
    it will use the pip from that environment, otherwise uses the system pip.
    
    Args:
        package: Package name to install (e.g., "requests", "pandas==1.5.0", "git+https://github.com/user/repo.git")
        venv_path: Optional path to virtual environment - if provided, uses venv's pip instead of system pip
    
    Returns:
        CommandResult with success status, stdout/stderr output from pip, and return_code
    """
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
    """Execute a shell command and return its output and exit status.
    
    This tool runs shell commands with configurable working directory and timeout.
    Use with caution as it can execute arbitrary system commands.
    
    Args:
        command: Shell command to execute (e.g., "ls -la", "git status", "python --version")
        cwd: Optional working directory to run command in - supports ~ expansion and relative paths
        timeout: Command timeout in seconds (default: 30) - command will be killed if it runs longer
    
    Returns:
        CommandResult with success status, stdout/stderr output, and return_code
    """
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
    """Execute a Python script with optional arguments and virtual environment.
    
    This tool runs Python scripts using either the system Python or a virtual environment's
    Python interpreter. Supports passing command-line arguments to the script.
    
    Args:
        script_path: Path to the Python script file to execute - supports ~ expansion and relative paths
        args: Optional list of command-line arguments to pass to the script (e.g., ["--verbose", "input.txt"])
        venv: Optional path to virtual environment - if provided, uses venv's Python interpreter
        timeout: Script execution timeout in seconds (default: 60) - script will be killed if it runs longer
    
    Returns:
        CommandResult with success status, stdout/stderr output, and return_code from script execution
    """
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
    """Initialize a new debug session with organized artifact storage.
    
    This tool creates a structured debug environment within a project directory,
    setting up organized folders for storing debug artifacts, responses, code samples,
    analysis results, and logs. Each session gets a unique ID for isolation.
    
    Args:
        project_path: Root path of the project to debug - supports ~ expansion and relative paths (e.g., "~/my_project", "./current_project")
    
    Returns:
        Dict with success status, unique session_id, debug_directory path (debug_artifacts/sessions/{id}), and project_path confirmation
    """
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
    """Retrieve all artifacts and metadata for an active debug session.
    
    This tool lists all files and artifacts that have been collected during
    a debug session, organized by type (responses, code, analysis, logs).
    
    Args:
        session_id: Unique session identifier from setup_debug_session (8-character string)
    
    Returns:
        Dict with success status, complete session metadata, and artifacts dictionary organized by type with file lists
    """
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
    """Extract and categorize CSS selectors, XPath expressions, and Playwright selectors from Python code.
    
    This specialized tool analyzes Python code (particularly web automation and testing code)
    to identify and categorize different types of selectors used for web element identification.
    Useful for debugging selector issues or analyzing automation test code.
    
    Args:
        code: Python source code to analyze for selectors (typically web automation or testing code)
    
    Returns:
        SelectorAnalysis with selectors array (all unique selectors found), selector_types dict (categorized by css/xpath/playwright), and analysis summary string
    """
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
    parser.add_argument("--working-dir", help="Set initial working directory for file operations")
    
    args = parser.parse_args()
    
    # Set initial working directory if provided
    if args.working_dir:
        global current_working_directory
        try:
            initial_path = Path(args.working_dir).expanduser().resolve()
            if initial_path.exists() and initial_path.is_dir():
                current_working_directory = initial_path
                print(f"Working directory set to: {current_working_directory}")
            else:
                print(f"Warning: Working directory does not exist or is not a directory: {args.working_dir}")
        except Exception as e:
            print(f"Warning: Could not set working directory: {e}")
    
    if args.transport == "stdio":
        mcp.run()
    else:
        # HTTP transport using StreamableHTTP
        import uvicorn
        app = mcp.streamable_http_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()