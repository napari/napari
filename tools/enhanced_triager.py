#!/usr/bin/env python3
"""
Enhanced GitHub Issue Triager for napari using MCP GitHub integration.

This script integrates with the GitHub MCP server to provide real-time issue
triaging capabilities for the napari repository.
"""

import json
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import the core triaging functionality
try:
    from issue_triager import IssueTriage
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from issue_triager import IssueTriage


class NapariIssueTriager:
    """Enhanced issue triager with MCP GitHub integration."""
    
    def __init__(self):
        self.triager = IssueTriage()
        self.repo_owner = "napari"
        self.repo_name = "napari"
        
    def convert_github_issue_to_triager_format(self, github_issue: Dict) -> Dict:
        """Convert GitHub MCP issue format to our triager format."""
        return {
            'number': github_issue['number'],
            'title': github_issue['title'],
            'body': github_issue.get('body', ''),
            'labels': github_issue.get('labels', []),
            'updated_at': github_issue['updated_at'],
            'created_at': github_issue['created_at'],
            'user': github_issue.get('user', {}),
            'state': github_issue['state'],
            'comments': github_issue.get('comments', 0)
        }
    
    def analyze_issues_from_api_response(self, api_response: Dict) -> Dict[str, Any]:
        """Analyze issues from GitHub MCP API response."""
        if 'issues' not in api_response:
            return {'error': 'Invalid API response format'}
            
        issues = []
        for github_issue in api_response['issues']:
            # Skip pull requests
            if 'pull_request' in github_issue:
                continue
            converted_issue = self.convert_github_issue_to_triager_format(github_issue)
            issues.append(converted_issue)
        
        if not issues:
            return {'message': 'No issues found to analyze'}
            
        # Generate comprehensive analysis
        report = self.triager.generate_triage_report(issues)
        
        # Add detailed issue analysis
        detailed_analysis = []
        for issue in issues:
            analysis = self.triager.analyze_issue_content(issue)
            detailed_analysis.append({
                'issue': issue,
                'analysis': analysis,
                'suggested_actions': self._generate_suggested_actions(issue, analysis)
            })
        
        report['detailed_analysis'] = detailed_analysis
        report['api_metadata'] = api_response.get('pageInfo', {})
        
        return report
    
    def _generate_suggested_actions(self, issue: Dict, analysis: Dict) -> List[str]:
        """Generate suggested actions for an issue based on analysis."""
        actions = []
        
        # Suggest labels to add
        current_label_names = {label.get('name', '') for label in issue.get('labels', [])}
        
        if analysis['suggested_components']:
            missing_components = analysis['suggested_components'] - current_label_names
            if missing_components:
                actions.append(f"Add component labels: {', '.join(missing_components)}")
        
        if analysis['suggested_priority'] and f"priority-{analysis['suggested_priority']}" not in current_label_names:
            actions.append(f"Add priority label: priority-{analysis['suggested_priority']}")
        
        if analysis['is_bug'] and not any('bug' in label for label in current_label_names):
            actions.append("Add 'bug' label")
        
        if analysis['is_feature'] and not any(term in current_label_names for term in ['enhancement', 'feature']):
            actions.append("Add 'enhancement' label")
        
        # Suggest information requests
        if analysis['needs_environment_info']:
            actions.append("Request environment information (napari --info)")
        
        if analysis['needs_triage']:
            actions.append("Issue needs initial triage and labeling")
        
        # Suggest closing criteria for stale issues
        updated_at = datetime.fromisoformat(issue['updated_at'].replace('Z', '+00:00'))
        days_since_update = (datetime.now(updated_at.tzinfo) - updated_at).days
        
        if days_since_update > 90 and issue['comments'] == 0:
            actions.append("Consider closing - no activity for 90+ days with no comments")
        elif days_since_update > 180:
            actions.append("Consider closing - no activity for 180+ days")
        
        return actions
    
    def generate_triage_summary(self, report: Dict, output_format: str = 'text') -> str:
        """Generate a formatted triage summary."""
        if 'error' in report:
            return f"Error: {report['error']}"
        
        if 'message' in report and 'detailed_analysis' not in report:
            return report['message']
        
        if output_format == 'json':
            return json.dumps(report, indent=2, default=str)
        
        # Text format
        lines = []
        lines.append("=== Napari Issue Triage Report ===")
        lines.append(f"Generated: {report.get('generated_at', 'unknown')}")
        lines.append(f"Total issues analyzed: {report.get('total_issues', 0)}")
        lines.append("")
        
        # Summary statistics
        lines.append("--- Summary Statistics ---")
        lines.append(f"Issues needing triage: {report.get('needs_triage', 0)}")
        lines.append(f"Bug reports missing env info: {report.get('needs_env_info', 0)}")
        lines.append("")
        
        # Component breakdown
        if report.get('component_breakdown'):
            lines.append("--- Component Breakdown ---")
            for component, count in sorted(report['component_breakdown'].items(), 
                                         key=lambda x: x[1], reverse=True):
                lines.append(f"{component}: {count}")
            lines.append("")
        
        # Priority breakdown
        if report.get('priority_breakdown'):
            lines.append("--- Priority Breakdown ---")
            for priority, count in sorted(report['priority_breakdown'].items(), 
                                        key=lambda x: x[1], reverse=True):
                lines.append(f"{priority}: {count}")
            lines.append("")
        
        # Recommendations
        if report.get('recommendations'):
            lines.append("--- Recommendations ---")
            for rec in report['recommendations']:
                lines.append(f"[{rec['type'].upper()}] {rec['message']}")
                if 'action' in rec:
                    lines.append(f"  Action: {rec['action']}")
                if 'issues' in rec:
                    lines.append("  Affected issues:")
                    for issue_ref in rec['issues']:
                        lines.append(f"    - #{issue_ref['number']}: {issue_ref['title'][:60]}...")
            lines.append("")
        
        # Detailed issue analysis (limited to most important)
        if report.get('detailed_analysis'):
            high_priority_issues = []
            needs_attention = []
            
            for item in report['detailed_analysis']:
                issue = item['issue']
                analysis = item['analysis']
                actions = item['suggested_actions']
                
                if analysis.get('suggested_priority') in ['critical', 'high']:
                    high_priority_issues.append(item)
                elif analysis.get('needs_triage') or actions:
                    needs_attention.append(item)
            
            # Show high priority issues first
            if high_priority_issues:
                lines.append("--- High Priority Issues ---")
                for item in high_priority_issues[:5]:  # Limit to top 5
                    issue = item['issue']
                    actions = item['suggested_actions']
                    lines.append(f"#{issue['number']}: {issue['title'][:70]}...")
                    if actions:
                        for action in actions[:3]:  # Limit actions
                            lines.append(f"  • {action}")
                lines.append("")
            
            # Show issues needing attention
            if needs_attention:
                lines.append("--- Issues Needing Attention ---")
                for item in needs_attention[:10]:  # Limit to top 10
                    issue = item['issue']
                    actions = item['suggested_actions']
                    if actions:
                        lines.append(f"#{issue['number']}: {issue['title'][:70]}...")
                        for action in actions[:2]:  # Limit actions
                            lines.append(f"  • {action}")
                lines.append("")
        
        return "\n".join(lines)
    
    def generate_label_suggestions_batch(self, issues_data: List[Dict]) -> Dict[int, List[str]]:
        """Generate label suggestions for a batch of issues."""
        suggestions = {}
        
        for issue_data in issues_data:
            issue = self.convert_github_issue_to_triager_format(issue_data)
            analysis = self.triager.analyze_issue_content(issue)
            
            current_labels = {label.get('name', '') for label in issue.get('labels', [])}
            suggested_labels = []
            
            # Add component suggestions
            for component in analysis['suggested_components']:
                if component not in current_labels:
                    suggested_labels.append(component)
            
            # Add priority suggestion
            if analysis['suggested_priority']:
                priority_label = f"priority-{analysis['suggested_priority']}"
                if priority_label not in current_labels:
                    suggested_labels.append(priority_label)
            
            # Add type suggestions
            if analysis['is_bug'] and not any('bug' in label for label in current_labels):
                suggested_labels.append('bug')
            
            if analysis['is_feature'] and not any(term in current_labels for term in ['enhancement', 'feature']):
                suggested_labels.append('enhancement')
            
            # Add special flags
            if analysis['needs_environment_info']:
                suggested_labels.append('needs-info')
            
            if suggested_labels:
                suggestions[issue['number']] = suggested_labels
        
        return suggestions


def main():
    """Main CLI interface for the enhanced triager."""
    parser = argparse.ArgumentParser(
        description="Enhanced GitHub Issue Triager for napari",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_triager.py --demo                    # Run demo with sample data
  python enhanced_triager.py --format json            # Output in JSON format
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with sample issues')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format (text or json)')
    parser.add_argument('--output', '-o',
                       help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    triager = NapariIssueTriager()
    
    if args.demo:
        # Use sample data from recent napari issues
        sample_api_response = {
            "issues": [
                {
                    "number": 8234,
                    "state": "OPEN",
                    "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
                    "body": "Creating a temp environment to launch napari with Qt6 does not work (found through debugging a different setup for someone): `uvx --with \"napari[pyqt6]\" napari`. Using Qt5 works as expected. Error message shows QtBindingsNotFoundError: No Qt bindings could be found.",
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
                    "body": "When using napari 0.6.3 or later in combination with numpy 2.2.6, created shapes will show some strange behavior: shapes will not be drawn in the right location, selecting a shape and moving it will visually affect another shape",
                    "user": {"login": "thopp-tudelft"},
                    "labels": [{"name": "bug"}],
                    "comments": 4,
                    "created_at": "2025-08-25T12:25:47Z",
                    "updated_at": "2025-08-26T05:41:04Z"
                },
                {
                    "number": 8225,
                    "state": "OPEN",
                    "title": "Consider using `array-api-compat` as comptybility backend for different array backends", 
                    "body": "Check if we could use array-api-compat as a compat layer to reduce code required to handle different array backends.",
                    "user": {"login": "Czaki"},
                    "labels": [{"name": "task"}],
                    "comments": 3,
                    "created_at": "2025-08-20T09:00:06Z",
                    "updated_at": "2025-08-22T13:54:44Z"
                }
            ],
            "pageInfo": {"hasNextPage": True, "totalCount": 1029}
        }
        
        print("Running napari issue triaging demo...")
        report = triager.analyze_issues_from_api_response(sample_api_response)
        output = triager.generate_triage_summary(report, args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)
    else:
        print("Enhanced Issue Triager for napari")
        print("Use --demo to run with sample data")
        print("Or integrate this script with GitHub MCP tools")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())