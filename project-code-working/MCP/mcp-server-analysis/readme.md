install uv

uv pip install -r pyproject.toml

uv run main.py




 % cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
{
    "mcpServers": {
        "mcp-server": {
            "command": "/Users/gshiva/.local/bin/uv",
            "args": [
                "--directory",
                "/Users/gshiva/mcp/mcp-server/mcp-server-example",
                "run",
                "main.py"
            ]
        }
    }
}
