"""Setup script for debug-mcp-server."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="debug-mcp-server",
    version="0.1.0",
    author="Alex Williamson",
    author_email="your.email@example.com",  # Update with your email
    description="Debug MCP Server for scraper development workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexwilliamson/debug-mcp-server",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "debug-mcp=debug_mcp.server:main",
        ],
    },
    keywords="mcp, debug, scraping, development, automation, ai",
    project_urls={
        "Bug Reports": "https://github.com/alexwilliamson/debug-mcp-server/issues",
        "Source": "https://github.com/alexwilliamson/debug-mcp-server",
        "Documentation": "https://github.com/alexwilliamson/debug-mcp-server#readme",
    },
)