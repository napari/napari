#!/usr/bin/env python3
"""
Quick Setup and Demo for Napari Issue Triaging

This script provides a quick way to set up and demonstrate
the napari issue triaging system.
"""

import subprocess
import sys
from pathlib import Path


def run_demo():
    """Run a complete demonstration of the triaging system."""
    
    print("🚀 Napari Issue Triaging System Setup & Demo")
    print("=" * 50)
    print()
    
    tools_dir = Path(__file__).parent
    
    # Check if all required files exist
    required_files = [
        'issue_triager.py',
        'enhanced_triager.py', 
        'triage_workflow.py',
        'live_triager.py',
        'test_triaging.py',
        'real_issue_analysis.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not (tools_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return 1
    
    print("✅ All triaging system files found")
    print()
    
    # Run tests first
    print("1. Running validation tests...")
    try:
        result = subprocess.run([sys.executable, 'test_triaging.py'], 
                              cwd=tools_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ All tests passed")
        else:
            print(f"   ❌ Tests failed: {result.stderr}")
            return 1
    except Exception as e:
        print(f"   ❌ Error running tests: {e}")
        return 1
    
    print()
    print("2. Running workflow demonstration...")
    try:
        result = subprocess.run([sys.executable, 'triage_workflow.py'], 
                              cwd=tools_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Workflow demo completed")
            # Show a snippet of the output
            lines = result.stdout.split('\n')
            summary_lines = [line for line in lines if 'Summary:' in line or 'issues need attention' in line]
            if summary_lines:
                print(f"   📊 {summary_lines[-1].strip()}")
        else:
            print(f"   ❌ Workflow demo failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error running workflow: {e}")
    
    print()
    print("3. Analyzing real napari issues...")
    try:
        result = subprocess.run([sys.executable, 'real_issue_analysis.py'], 
                              cwd=tools_dir, capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Real issue analysis completed")
            # Extract key metrics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Issues analyzed:' in line or 'Label suggestions:' in line or 'Need attention:' in line:
                    print(f"   📈 {line.strip()}")
        else:
            print(f"   ❌ Real analysis failed: {result.stderr}")
    except Exception as e:
        print(f"   ❌ Error running real analysis: {e}")
    
    print()
    print("🎯 SETUP COMPLETE!")
    print()
    print("📚 Available Commands:")
    commands = [
        "python tools/issue_triager.py --help          # Core triaging tool",
        "python tools/triage_workflow.py               # Complete workflow demo", 
        "python tools/real_issue_analysis.py           # Analyze real issues",
        "python tools/github_integration_demo.py       # GitHub MCP integration guide",
        "python tools/test_triaging.py                 # Run validation tests"
    ]
    
    for cmd in commands:
        print(f"  {cmd}")
    
    print()
    print("🔗 Integration Options:")
    integrations = [
        "• GitHub Actions workflow for automated triaging",
        "• Slack/Discord bot for maintainer notifications",
        "• Web dashboard for visual issue management", 
        "• CLI tool for manual triaging sessions",
        "• Jupyter notebook for interactive analysis"
    ]
    
    for integration in integrations:
        print(f"  {integration}")
    
    print()
    print("📖 Documentation: tools/README_TRIAGING.md")
    
    return 0


if __name__ == '__main__':
    sys.exit(run_demo())