#!/usr/bin/env python3
"""
Live GitHub MCP Integration for Napari Issue Triaging

This script demonstrates how to use the actual GitHub MCP server tools
available in this environment with the napari issue triaging system.
"""

import json
import sys
from pathlib import Path

# Add tools directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_triager import NapariIssueTriager


def live_mcp_demonstration():
    """
    Demonstrate integration with GitHub MCP server tools.
    
    This would be called by the environment that has access to:
    - github-mcp-server-list_issues
    - github-mcp-server-get_issue  
    - github-mcp-server-search_issues
    etc.
    """
    
    print("🔴 Live GitHub MCP Server Integration")
    print("=" * 40)
    print()
    
    # Sample of what the MCP tools would return
    # This is the actual structure from github-mcp-server-list_issues
    mcp_issues_response = {
        "issues": [
            {
                "id": 3364623284,
                "number": 8234,
                "state": "OPEN",
                "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
                "body": "Creating a temp environment to launch napari with Qt6 does not work. Error: qtpy.QtBindingsNotFoundError: No Qt bindings could be found",
                "user": {"login": "TimMonko"},
                "labels": [{"name": "bug", "description": "Something isn't working as expected"}],
                "comments": 3,
                "created_at": "2025-08-28T20:44:04Z",
                "updated_at": "2025-08-28T21:00:17Z"
            },
            {
                "id": 3351650067,
                "number": 8230,
                "state": "OPEN",
                "title": "bugs with stacked shapes layer since 0.6.3",
                "body": "When using napari 0.6.3 or later with numpy 2.2.6, shapes layer has rendering issues. Environment info included: napari: 0.6.4, Platform: Linux, Python: 3.13.7, Qt: 6.9.0, NumPy: 2.2.6",
                "user": {"login": "thopp-tudelft"},
                "labels": [{"name": "bug", "description": "Something isn't working as expected"}],
                "comments": 4,
                "created_at": "2025-08-25T12:25:47Z",
                "updated_at": "2025-08-26T05:41:04Z"
            }
        ],
        "pageInfo": {
            "endCursor": "cursor123",
            "hasNextPage": True,
            "hasPreviousPage": False,
            "startCursor": "cursor456",
            "totalCount": 1029
        }
    }
    
    triager = NapariIssueTriager()
    
    print("📡 Simulating MCP GitHub Integration:")
    print("   github-mcp-server-list_issues(owner='napari', repo='napari', state='OPEN')")
    print()
    
    # Process the MCP response
    print("🔍 Processing GitHub MCP response...")
    report = triager.analyze_issues_from_api_response(mcp_issues_response)
    
    print("📊 Analysis Results:")
    print(f"   • Issues analyzed: {report['total_issues']}")
    print(f"   • Need triage: {report['needs_triage']}")
    print(f"   • Missing env info: {report['needs_env_info']}")
    print()
    
    # Show detailed analysis for each issue
    print("🏷️ Issue-by-Issue Analysis:")
    print("-" * 30)
    
    for item in report.get('detailed_analysis', []):
        issue = item['issue']
        analysis = item['analysis']
        actions = item['suggested_actions']
        
        print(f"\n#{issue['number']}: {issue['title'][:50]}...")
        
        current_labels = [label.get('name', '') for label in issue.get('labels', [])]
        print(f"   Current labels: {current_labels}")
        
        # Show suggested components
        if analysis['suggested_components']:
            print(f"   Suggested components: {list(analysis['suggested_components'])}")
        
        # Show priority
        if analysis['suggested_priority']:
            print(f"   Suggested priority: {analysis['suggested_priority']}")
        
        # Show type classification
        types = []
        if analysis['is_bug']:
            types.append('bug')
        if analysis['is_feature']:
            types.append('feature')
        if types:
            print(f"   Type: {', '.join(types)}")
        
        # Show immediate actions needed
        if actions:
            print("   Actions needed:")
            for action in actions[:3]:  # Show top 3 actions
                print(f"     • {action}")
        
        # Special flags
        flags = []
        if analysis['needs_triage']:
            flags.append('🔍 NEEDS_TRIAGE')
        if analysis['needs_environment_info']:
            flags.append('⚠️ MISSING_ENV_INFO')
        
        if flags:
            print(f"   Flags: {' '.join(flags)}")
    
    # Show how to integrate with other MCP tools
    print(f"\n🔧 Additional MCP Integration Opportunities:")
    
    integration_examples = [
        {
            'tool': 'github-mcp-server-get_issue',
            'use_case': 'Get detailed info for specific high-priority issues',
            'example': 'github-mcp-server-get_issue(owner="napari", repo="napari", issue_number=8234)'
        },
        {
            'tool': 'github-mcp-server-get_issue_comments', 
            'use_case': 'Analyze comment history for context',
            'example': 'github-mcp-server-get_issue_comments(owner="napari", repo="napari", issue_number=8234)'
        },
        {
            'tool': 'github-mcp-server-search_issues',
            'use_case': 'Find related issues or duplicates',
            'example': 'github-mcp-server-search_issues(query="Qt6 installation napari")'
        }
    ]
    
    for example in integration_examples:
        print(f"\n• {example['tool']}")
        print(f"  Use case: {example['use_case']}")
        print(f"  Example: {example['example']}")
    
    # Export comprehensive dashboard
    print(f"\n📄 Exporting dashboard...")
    
    # Create a simple dashboard from the report data
    from datetime import datetime
    dashboard = {
        "generated_at": datetime.now().isoformat(),
        "repository": "napari/napari",
        "summary": {
            "issues_analyzed": report.get('total_issues', 0),
            "needing_triage": report.get('needs_triage', 0),
            "missing_env_info": report.get('needs_env_info', 0)
        },
        "report": report,
        "quick_actions": []
    }
    
    # Add quick actions
    for item in report.get('detailed_analysis', []):
        if item['analysis'].get('suggested_priority') in ['critical', 'high']:
            dashboard['quick_actions'].append({
                'type': 'urgent',
                'issue_number': item['issue']['number'],
                'title': item['issue']['title'],
                'priority': item['analysis']['suggested_priority']
            })
    
    with open("/tmp/napari_mcp_dashboard.json", 'w') as f:
        json.dump(dashboard, f, indent=2, default=str)
    
    print(f"✅ Dashboard exported with {len(dashboard.get('quick_actions', []))} action items")
    
    # Show sample quick actions
    quick_actions = dashboard.get('quick_actions', [])
    if quick_actions:
        print(f"\n📋 Quick Actions for Maintainers:")
        for action in quick_actions:
            if action['type'] == 'urgent':
                print(f"   🔥 URGENT: Issue #{action['issue_number']} ({action['priority']})")
            elif action['type'] == 'batch_action':
                print(f"   📧 BATCH: {action['description']}")
                print(f"      Affects issues: {action['issues']}")
    
    print(f"\n🎉 Live MCP integration demonstration complete!")
    print(f"🔗 Dashboard file: /tmp/napari_mcp_dashboard.json")
    
    return 0


def show_integration_code():
    """Show example integration code."""
    
    print("\n💻 MCP Integration Code Example:")
    print("-" * 35)
    
    code_example = '''
# Real integration with GitHub MCP server tools:

def triage_napari_issues_with_mcp():
    """Complete triaging workflow using MCP tools."""
    
    from tools.enhanced_triager import NapariIssueTriager
    
    # Initialize triager
    triager = NapariIssueTriager()
    
    # Step 1: Fetch open issues using MCP
    issues_response = github_mcp_server_list_issues(
        owner="napari",
        repo="napari", 
        state="OPEN",
        perPage=50
    )
    
    # Step 2: Analyze issues with our triaging system
    report = triager.analyze_issues_from_api_response(issues_response)
    
    # Step 3: Generate actionable insights
    summary = triager.generate_triage_summary(report)
    print(summary)
    
    # Step 4: Get detailed analysis for high-priority issues
    for item in report['detailed_analysis']:
        if item['analysis']['suggested_priority'] in ['critical', 'high']:
            issue_num = item['issue']['number']
            
            # Fetch additional details
            issue_details = github_mcp_server_get_issue(
                owner="napari",
                repo="napari",
                issue_number=issue_num
            )
            
            # Get comment history for context
            comments = github_mcp_server_get_issue_comments(
                owner="napari", 
                repo="napari",
                issue_number=issue_num
            )
            
            # Enhanced analysis with comments
            # ... additional processing logic
    
    # Step 5: Export dashboard for maintainers
    dashboard = triager.export_triage_dashboard()
    
    return dashboard
'''
    
    print(code_example)


if __name__ == '__main__':
    result = live_mcp_demonstration()
    if result == 0:
        show_integration_code()
    sys.exit(result)