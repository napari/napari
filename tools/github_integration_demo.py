#!/usr/bin/env python3
"""
Real GitHub Integration Example

This script demonstrates how to use the GitHub MCP server tools 
with the napari issue triaging system for live issue management.

Note: This requires the GitHub MCP server tools to be available
in the environment.
"""

import json
import sys
from pathlib import Path

# Add tools directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_triager import NapariIssueTriager


def demonstrate_real_github_integration():
    """
    Demonstrate real GitHub integration.
    
    This function shows how you would integrate the triaging system
    with actual GitHub MCP server calls.
    """
    
    print("🔴 Real GitHub Integration Demo")
    print("=" * 40)
    print()
    
    triager = NapariIssueTriager()
    
    print("📡 This example shows how to integrate with GitHub MCP tools:")
    print()
    
    # Show the MCP tool call pattern
    integration_example = '''
# Example integration with GitHub MCP tools:

# 1. Fetch open issues
issues_response = github-mcp-server-list_issues(
    owner="napari",
    repo="napari", 
    state="OPEN",
    perPage=50
)

# 2. Analyze with our triager
report = triager.analyze_issues_from_api_response(issues_response)

# 3. Generate actionable summary
summary = triager.generate_triage_summary(report)

# 4. Get specific issue details if needed
issue_details = github-mcp-server-get_issue(
    owner="napari",
    repo="napari", 
    issue_number=1234
)

# 5. Analyze specific issue
analysis = triager.triager.analyze_issue_content(issue_details)

# 6. Export comprehensive dashboard
dashboard = triager.export_triage_dashboard("/tmp/dashboard.json")
'''
    
    print(integration_example)
    
    print("🔧 Available MCP Tools for Integration:")
    mcp_tools = [
        "github-mcp-server-list_issues - List issues with filters",
        "github-mcp-server-get_issue - Get detailed issue information", 
        "github-mcp-server-get_issue_comments - Get issue comments for context",
        "github-mcp-server-search_issues - Search issues by criteria",
        "github-mcp-server-list_code_scanning_alerts - Security issue detection",
        "github-mcp-server-list_secret_scanning_alerts - Secret exposure detection"
    ]
    
    for tool in mcp_tools:
        print(f"  • {tool}")
    
    print()
    print("🎯 Triaging Workflow with MCP Tools:")
    
    workflow_steps = [
        "1. Fetch recent issues using list_issues",
        "2. For each issue, run content analysis",
        "3. Generate label suggestions based on keywords",
        "4. Identify issues needing maintainer attention", 
        "5. Check for missing environment information",
        "6. Generate priority-based action items",
        "7. Export dashboard for maintainer review",
        "8. Optionally auto-apply obvious labels (if permissions allow)"
    ]
    
    for step in workflow_steps:
        print(f"  {step}")
    
    print()
    print("📊 Sample Triaging Analysis:")
    
    # Demonstrate with a sample issue
    sample_issue = {
        "number": 8234,
        "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
        "body": "Creating a temp environment to launch napari with Qt6 does not work. Error: qtpy.QtBindingsNotFoundError: No Qt bindings could be found",
        "labels": [{"name": "bug"}],
        "user": {"login": "TimMonko"},
        "updated_at": "2025-08-28T21:00:17Z",
        "created_at": "2025-08-28T20:44:04Z",
        "comments": 3,
        "state": "OPEN"
    }
    
    analysis = triager.triager.analyze_issue_content(sample_issue)
    
    print(f"Issue: #{sample_issue['number']}")
    print(f"Title: {sample_issue['title']}")
    print(f"Current labels: {[label['name'] for label in sample_issue['labels']]}")
    print(f"Suggested components: {list(analysis['suggested_components'])}")
    print(f"Is bug: {analysis['is_bug']}")
    print(f"Needs env info: {analysis['needs_environment_info']}")
    
    # Generate suggestions
    current_labels = {label['name'] for label in sample_issue['labels']}
    suggestions = []
    
    for component in analysis['suggested_components']:
        if component not in current_labels:
            suggestions.append(component)
    
    if analysis['needs_environment_info']:
        suggestions.append('needs-info')
    
    print(f"Label suggestions: +{', '.join(suggestions)}")
    
    print()
    print("💡 Integration Benefits:")
    benefits = [
        "• Automatic component labeling based on content analysis",
        "• Priority assessment for faster triage",
        "• Missing information detection for bug reports",
        "• Batch processing for efficient maintainer workflows",
        "• Consistent labeling standards across the repository",
        "• Analytics and reporting for issue management insights"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    return 0


if __name__ == '__main__':
    sys.exit(demonstrate_real_github_integration())