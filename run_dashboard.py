#!/usr/bin/env python3
"""
Run AI Security Dashboard
=========================

Quick script to start the LangFuse-powered security dashboard
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🛡️  Starting AI Security Dashboard...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"])
    
    # Check if langfuse is installed
    try:
        import langfuse
        print("✅ LangFuse is available")
    except ImportError:
        print("⚠️  LangFuse not available. Dashboard will run without LangFuse integration.")
        print("   To enable LangFuse: pip install langfuse")
    
    print("\n🚀 Starting dashboard...")
    print("   Dashboard will open in your browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    # Run the dashboard
    dashboard_path = Path(__file__).parent / "src" / "langfuse_dashboard.py"
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])

if __name__ == "__main__":
    main() 