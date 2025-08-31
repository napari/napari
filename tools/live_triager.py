#!/usr/bin/env python3
"""
Live GitHub Issue Triaging for napari

This script integrates with GitHub MCP server tools to provide real-time
issue triaging for the napari repository.

Note: This script is designed to work with the GitHub MCP server tools
available in the environment. It demonstrates how the triaging system
would work with live GitHub data.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for local imports
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_triager import NapariIssueTriager
except ImportError:
    print("Error: enhanced_triager module not found")
    sys.exit(1)


class LiveNapariTriager:
    """Live GitHub issue triager using MCP tools."""
    
    def __init__(self):
        self.triager = NapariIssueTriager()
        self.repo_owner = "napari"
        self.repo_name = "napari"
    
    def triage_open_issues(self, max_issues: int = 20) -> dict:
        """
        Simulate triaging open issues.
        
        In a real implementation, this would call the GitHub MCP tools like:
        - github-mcp-server-list_issues
        - github-mcp-server-get_issue
        - github-mcp-server-get_issue_comments
        """
        
        print(f"🔍 Triaging up to {max_issues} open issues from napari/napari...")
        
        # This simulates what the MCP tool would return
        # In practice, you'd call: github-mcp-server-list_issues(owner="napari", repo="napari", state="OPEN")
        mock_api_response = {
            "issues": [
                {
                    "number": 8234,
                    "state": "OPEN",
                    "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
                    "body": "Creating a temp environment to launch napari with Qt6 does not work. Error with qtpy.QtBindingsNotFoundError: No Qt bindings could be found. This is an installation and Qt backend issue.",
                    "user": {"login": "TimMonko"},
                    "labels": [{"name": "bug"}],
                    "comments": 3,
                    "created_at": "2025-08-28T20:44:04Z",
                    "updated_at": "2025-08-28T21:00:17Z"
                },
                {
                    "number": 8230,
                    "state": "OPEN",
                    "title": "bugs with stacked shapes layer since 0.6.3",
                    "body": "Shapes layer rendering bugs with numpy 2.2.6. Shapes not drawn in right location, visual glitches when moving shapes. This affects the shapes layer specifically.",
                    "user": {"login": "thopp-tudelft"},
                    "labels": [{"name": "bug"}],
                    "comments": 4,
                    "created_at": "2025-08-25T12:25:47Z",
                    "updated_at": "2025-08-26T05:41:04Z"
                },
                {
                    "number": 8215,
                    "state": "OPEN",
                    "title": "Memory leak when opening large images",
                    "body": "Opening large images (>2GB) causes memory usage to continuously increase. Performance degrades over time. This is a critical performance issue.",
                    "user": {"login": "researcher1"},
                    "labels": [],
                    "comments": 0,
                    "created_at": "2025-08-10T14:20:00Z",
                    "updated_at": "2025-08-10T14:20:00Z"
                },
                {
                    "number": 8200,
                    "state": "OPEN", 
                    "title": "Add support for 3D volume rendering improvements",
                    "body": "Feature request to improve 3D volume rendering with better shaders and GPU acceleration. Would enhance the vispy rendering pipeline.",
                    "user": {"login": "viz_expert"},
                    "labels": [{"name": "enhancement"}],
                    "comments": 2,
                    "created_at": "2025-08-05T09:15:00Z",
                    "updated_at": "2025-08-12T16:30:00Z"
                }
            ],
            "pageInfo": {"hasNextPage": True, "totalCount": 1029}
        }
        
        # Analyze using our triager
        report = self.triager.analyze_issues_from_api_response(mock_api_response)
        return report
    
    def triage_recent_issues(self, days: int = 7) -> dict:
        """Triage issues updated in the last N days."""
        
        print(f"🔍 Triaging issues updated in the last {days} days...")
        
        # This would use: github-mcp-server-list_issues with since parameter
        # For demo, using recent issues
        recent_api_response = {
            "issues": [
                {
                    "number": 8234,
                    "state": "OPEN",
                    "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
                    "body": "Qt6 installation issue with uv tool. QtBindingsNotFoundError when using uvx commands.",
                    "user": {"login": "TimMonko"},
                    "labels": [{"name": "bug"}],
                    "comments": 3,
                    "created_at": "2025-08-28T20:44:04Z",
                    "updated_at": "2025-08-28T21:00:17Z"
                },
                {
                    "number": 8230,
                    "state": "OPEN",
                    "title": "bugs with stacked shapes layer since 0.6.3",
                    "body": "Shapes layer issues with numpy 2.2.6 - incorrect rendering and selection problems",
                    "user": {"login": "thopp-tudelft"},
                    "labels": [{"name": "bug"}],
                    "comments": 4,
                    "created_at": "2025-08-25T12:25:47Z",
                    "updated_at": "2025-08-26T05:41:04Z"
                }
            ]
        }
        
        report = self.triager.analyze_issues_from_api_response(recent_api_response)
        return report
    
    def export_triage_dashboard(self, output_file: str = "/tmp/napari_dashboard.json"):
        """Export a complete triaging dashboard."""
        
        # Get comprehensive data
        open_issues_report = self.triage_open_issues(max_issues=50)
        recent_issues_report = self.triage_recent_issues(days=7)
        
        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "repository": f"{self.repo_owner}/{self.repo_name}",
            "summary": {
                "open_issues_analyzed": open_issues_report.get('total_issues', 0),
                "recent_issues_analyzed": recent_issues_report.get('total_issues', 0),
                "total_needing_triage": open_issues_report.get('needs_triage', 0),
                "total_missing_env_info": open_issues_report.get('needs_env_info', 0)
            },
            "reports": {
                "open_issues": open_issues_report,
                "recent_activity": recent_issues_report
            },
            "quick_actions": self._generate_quick_actions(open_issues_report)
        }
        
        with open(output_file, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)
        
        return dashboard
    
    def _generate_quick_actions(self, report: dict) -> list:
        """Generate quick action items for maintainers."""
        actions = []
        
        # High priority items first
        for item in report.get('detailed_analysis', []):
            issue = item['issue']
            analysis = item['analysis']
            suggested_actions = item['suggested_actions']
            
            if analysis.get('suggested_priority') in ['critical', 'high']:
                actions.append({
                    'type': 'urgent',
                    'issue_number': issue['number'],
                    'title': issue['title'],
                    'priority': analysis['suggested_priority'],
                    'actions': suggested_actions[:3]  # Top 3 actions
                })
        
        # Issues needing environment info
        env_info_needed = []
        for item in report.get('detailed_analysis', []):
            if item['analysis'].get('needs_environment_info'):
                env_info_needed.append(item['issue']['number'])
        
        if env_info_needed:
            actions.append({
                'type': 'batch_action',
                'description': 'Request environment info from bug reporters',
                'issues': env_info_needed,
                'template': "Hi! Thanks for the bug report. Could you please run `napari --info` and share the output? This will help us debug the issue."
            })
        
        return actions


def main():
    """Main demonstration function."""
    print("🚀 Live Napari Issue Triaging System")
    print("=====================================")
    
    triager = LiveNapariTriager()
    
    # Demonstrate different triaging workflows
    print("\n1. Triaging Open Issues")
    print("-" * 30)
    open_report = triager.triage_open_issues(max_issues=10)
    open_summary = triager.triager.generate_triage_summary(open_report, 'text')
    
    # Print a condensed version
    lines = open_summary.split('\n')
    summary_lines = [line for line in lines[:15]]  # First 15 lines
    print('\n'.join(summary_lines))
    
    print("\n2. Triaging Recent Activity")
    print("-" * 30)
    recent_report = triager.triage_recent_issues(days=7)
    recent_summary = triager.triager.generate_triage_summary(recent_report, 'text')
    
    # Print key metrics
    print(f"Recent issues analyzed: {recent_report.get('total_issues', 0)}")
    print(f"Needing attention: {recent_report.get('needs_triage', 0)}")
    
    print("\n3. Exporting Dashboard")
    print("-" * 30)
    dashboard = triager.export_triage_dashboard()
    print(f"✅ Dashboard exported with {len(dashboard.get('quick_actions', []))} quick action items")
    
    # Show sample quick actions
    quick_actions = dashboard.get('quick_actions', [])
    if quick_actions:
        print("\n📋 Quick Actions for Maintainers:")
        for action in quick_actions[:3]:  # Show first 3
            if action['type'] == 'urgent':
                print(f"  🔥 URGENT: Issue #{action['issue_number']} - {action['title'][:50]}...")
                print(f"     Priority: {action['priority']}")
            elif action['type'] == 'batch_action':
                print(f"  📧 BATCH: {action['description']}")
                print(f"     Affects {len(action['issues'])} issues: {action['issues']}")
    
    print("\n4. Integration Points")
    print("-" * 30)
    print("🔗 This tool can be integrated with:")
    print("   • GitHub Actions for automated triaging")
    print("   • Slack/Discord bots for maintainer notifications") 
    print("   • Web dashboard for visual issue management")
    print("   • PR comment automation for missing info requests")
    
    print("\n✨ Demo completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())