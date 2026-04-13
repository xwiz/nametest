#!/usr/bin/env python3
"""
SRM API Server

Run the FastAPI server for programmatic access to SRM.

Usage:
    python api_server.py
    python api_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import uvicorn

from srm.api import app


def main():
    ap = argparse.ArgumentParser(description="SRM REST API Server")
    ap.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    ap.add_argument("--port", type=int, default=8000, help="Port to bind to")
    ap.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = ap.parse_args()
    
    print(f"Starting SRM API server on http://{args.host}:{args.port}")
    print("API documentation available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "srm.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
