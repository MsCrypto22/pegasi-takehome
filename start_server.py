#!/usr/bin/env python3
"""
Simple script to start the Adaptive MCP Server
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.adaptive_mcp_server import AdaptiveMCPServer

async def main():
    server = AdaptiveMCPServer(host="localhost", port=8000)
    print("Starting Adaptive MCP Server on localhost:8000")
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main()) 