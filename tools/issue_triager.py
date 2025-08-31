#!/usr/bin/env python3
"""
GitHub Issue Triager for napari

This tool helps triage GitHub issues by:
1. Fetching issues from the napari/napari repository
2. Analyzing issue content and suggesting appropriate labels
3. Categorizing issues by type, priority, and component
4. Generating summary reports for maintainers
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

try:
    import requests
except ImportError:
    print("Error: requests library is required. Install with: pip install requests")
    sys.exit(1)


class IssueTriage:
    """GitHub Issue Triage Assistant for napari."""
    
    # Base GitHub API URL
    BASE_URL = "https://api.github.com"
    REPO = "napari/napari"
    
    # Label mappings based on napari's existing labeling system
    COMPONENT_KEYWORDS = {
        'qt': ['qt', 'gui', 'widget', 'dialog', 'window', 'menu', 'toolbar'],
        'vispy': ['vispy', 'opengl', 'rendering', 'canvas', 'shader', 'gpu'],
        'layers': ['layer', 'image', 'labels', 'points', 'shapes', 'surface', 'tracks', 'vectors'],
        'io': ['read', 'write', 'save', 'load', 'file', 'import', 'export', 'format'],
        'plugins': ['plugin', 'npe2', 'extension', 'napari-plugin-engine'],
        'performance': ['slow', 'performance', 'memory', 'speed', 'optimization', 'lag'],
        'docs': ['documentation', 'docs', 'tutorial', 'example', 'guide'],
        'tests': ['test', 'testing', 'pytest', 'ci', 'continuous integration'],
        'installation': ['install', 'pip', 'conda', 'setup', 'environment', 'dependency'],
        'preferences': ['settings', 'preferences', 'config', 'configuration']
    }
    
    PRIORITY_KEYWORDS = {
        'critical': ['crash', 'segfault', 'data loss', 'critical', 'urgent', 'blocks'],
        'high': ['major', 'important', 'regression', 'breaking', 'blocker'],
        'medium': ['enhancement', 'feature', 'improvement'],
        'low': ['minor', 'cleanup', 'refactor', 'documentation']
    }
    
    BUG_KEYWORDS = [
        'bug', 'error', 'exception', 'crash', 'fail', 'broken', 'issue', 
        'problem', 'unexpected', 'wrong', 'incorrect', 'traceback'
    ]
    
    FEATURE_KEYWORDS = [
        'feature', 'enhancement', 'improvement', 'add', 'support', 'implement', 
        'request', 'would be nice', 'could we', 'can we'
    ]
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the triage tool.
        
        Parameters
        ----------
        token : str, optional
            GitHub personal access token. If not provided, will try to get from
            environment variable GITHUB_TOKEN.
        """
        self.token = token or os.environ.get('GITHUB_TOKEN')
        self.headers = {}
        if self.token:
            self.headers['Authorization'] = f'token {self.token}'
        self.headers['Accept'] = 'application/vnd.github.v3+json'
        
    def fetch_issues(self, state: str = 'open', labels: Optional[str] = None, 
                    since: Optional[str] = None, per_page: int = 100,
                    max_pages: int = 5) -> List[Dict]:
        """Fetch issues from GitHub API.
        
        Parameters
        ----------
        state : str, default 'open'
            State of issues to fetch ('open', 'closed', 'all').
        labels : str, optional
            Comma-separated list of labels to filter by.
        since : str, optional
            ISO 8601 datetime string to fetch issues updated since.
        per_page : int, default 100
            Number of issues per page.
        max_pages : int, default 5
            Maximum number of pages to fetch.
            
        Returns
        -------
        List[Dict]
            List of issue dictionaries from GitHub API.
        """
        issues = []
        page = 1
        
        while page <= max_pages:
            params = {
                'state': state,
                'per_page': per_page,
                'page': page,
                'sort': 'updated',
                'direction': 'desc'
            }
            
            if labels:
                params['labels'] = labels
            if since:
                params['since'] = since
                
            url = f"{self.BASE_URL}/repos/{self.REPO}/issues"
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                page_issues = response.json()
                if not page_issues:
                    break
                    
                # Filter out pull requests (GitHub API includes PRs in issues)
                page_issues = [issue for issue in page_issues if 'pull_request' not in issue]
                issues.extend(page_issues)
                
                print(f"Fetched page {page} ({len(page_issues)} issues)")
                page += 1
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching issues: {e}")
                break
                
        return issues
    
    def analyze_issue_content(self, issue: Dict) -> Dict[str, any]:
        """Analyze issue content and suggest labels and categorization.
        
        Parameters
        ----------
        issue : Dict
            Issue dictionary from GitHub API.
            
        Returns
        -------
        Dict[str, any]
            Analysis results including suggested labels and categories.
        """
        title = issue.get('title', '').lower()
        body = issue.get('body', '').lower() if issue.get('body') else ''
        current_labels = {label['name'] for label in issue.get('labels', [])}
        
        content = f"{title} {body}"
        
        analysis = {
            'suggested_components': set(),
            'suggested_priority': None,
            'is_bug': False,
            'is_feature': False,
            'confidence_scores': {},
            'current_labels': current_labels,
            'needs_triage': len(current_labels) == 0 or 'triage' in current_labels
        }
        
        # Component analysis
        for component, keywords in self.COMPONENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                analysis['suggested_components'].add(component)
                analysis['confidence_scores'][component] = score
        
        # Priority analysis
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                if not analysis['suggested_priority'] or score > analysis['confidence_scores'].get(f'priority_{priority}', 0):
                    analysis['suggested_priority'] = priority
                analysis['confidence_scores'][f'priority_{priority}'] = score
        
        # Bug vs Feature analysis
        bug_score = sum(1 for keyword in self.BUG_KEYWORDS if keyword in content)
        feature_score = sum(1 for keyword in self.FEATURE_KEYWORDS if keyword in content)
        
        analysis['is_bug'] = bug_score > 0
        analysis['is_feature'] = feature_score > 0
        analysis['confidence_scores']['bug'] = bug_score
        analysis['confidence_scores']['feature'] = feature_score
        
        # Check for missing environment info (common triage need)
        has_env_info = any(keyword in body for keyword in ['napari:', 'python:', 'platform:', 'qt:'])
        analysis['needs_environment_info'] = analysis['is_bug'] and not has_env_info
        
        return analysis
    
    def generate_triage_report(self, issues: List[Dict]) -> Dict[str, any]:
        """Generate a comprehensive triage report.
        
        Parameters
        ----------
        issues : List[Dict]
            List of issues to analyze.
            
        Returns
        -------
        Dict[str, any]
            Triage report with statistics and recommendations.
        """
        report = {
            'total_issues': len(issues),
            'needs_triage': 0,
            'needs_env_info': 0,
            'component_breakdown': defaultdict(int),
            'priority_breakdown': defaultdict(int),
            'type_breakdown': defaultdict(int),
            'label_coverage': defaultdict(int),
            'recommendations': [],
            'recent_activity': [],
            'generated_at': datetime.now().isoformat()
        }
        
        unlabeled_issues = []
        high_priority_unlabeled = []
        stale_issues = []
        one_week_ago = datetime.now() - timedelta(days=7)
        
        for issue in issues:
            analysis = self.analyze_issue_content(issue)
            
            # Basic statistics
            if analysis['needs_triage']:
                report['needs_triage'] += 1
                unlabeled_issues.append(issue)
                
            if analysis['needs_environment_info']:
                report['needs_env_info'] += 1
            
            # Component breakdown
            for component in analysis['suggested_components']:
                report['component_breakdown'][component] += 1
                
            # Priority breakdown
            if analysis['suggested_priority']:
                report['priority_breakdown'][analysis['suggested_priority']] += 1
                
            # Type breakdown
            if analysis['is_bug']:
                report['type_breakdown']['bug'] += 1
            if analysis['is_feature']:
                report['type_breakdown']['feature'] += 1
                
            # Label coverage
            for label in analysis['current_labels']:
                report['label_coverage'][label] += 1
                
            # Check for stale issues (no activity in 30+ days)
            updated_at = datetime.fromisoformat(issue['updated_at'].replace('Z', '+00:00'))
            if updated_at < datetime.now().replace(tzinfo=updated_at.tzinfo) - timedelta(days=30):
                stale_issues.append(issue)
                
            # Check for high-priority issues without labels
            if analysis['suggested_priority'] in ['critical', 'high'] and analysis['needs_triage']:
                high_priority_unlabeled.append(issue)
                
            # Recent activity
            if updated_at > one_week_ago.replace(tzinfo=updated_at.tzinfo):
                report['recent_activity'].append({
                    'number': issue['number'],
                    'title': issue['title'],
                    'updated_at': issue['updated_at'],
                    'author': issue['user']['login']
                })
        
        # Generate recommendations
        if high_priority_unlabeled:
            report['recommendations'].append({
                'type': 'urgent',
                'message': f"{len(high_priority_unlabeled)} high-priority issues need immediate triage",
                'issues': [{'number': i['number'], 'title': i['title']} for i in high_priority_unlabeled[:5]]
            })
            
        if report['needs_env_info'] > 0:
            report['recommendations'].append({
                'type': 'info_needed',
                'message': f"{report['needs_env_info']} bug reports are missing environment information",
                'action': "Ask reporters to run 'napari --info' and provide output"
            })
            
        if stale_issues:
            report['recommendations'].append({
                'type': 'stale',
                'message': f"{len(stale_issues)} issues haven't been updated in 30+ days",
                'action': "Consider closing stale issues or pinging for updates"
            })
            
        return report
    
    def suggest_labels_for_issue(self, issue_number: int) -> Dict[str, any]:
        """Suggest labels for a specific issue.
        
        Parameters
        ----------
        issue_number : int
            GitHub issue number.
            
        Returns
        -------
        Dict[str, any]
            Detailed analysis and label suggestions for the issue.
        """
        url = f"{self.BASE_URL}/repos/{self.REPO}/issues/{issue_number}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            issue = response.json()
            
            if 'pull_request' in issue:
                return {'error': 'This is a pull request, not an issue'}
                
            analysis = self.analyze_issue_content(issue)
            
            # Format suggestions
            suggestions = {
                'issue_number': issue_number,
                'title': issue['title'],
                'current_labels': list(analysis['current_labels']),
                'suggested_labels': [],
                'reasoning': [],
                'confidence': {}
            }
            
            # Add component suggestions
            for component in analysis['suggested_components']:
                suggestions['suggested_labels'].append(component)
                score = analysis['confidence_scores'][component]
                suggestions['confidence'][component] = score / 10  # Normalize to 0-1
                suggestions['reasoning'].append(f"Component '{component}' suggested based on {score} keyword matches")
            
            # Add priority suggestion
            if analysis['suggested_priority']:
                priority_label = f"priority-{analysis['suggested_priority']}"
                suggestions['suggested_labels'].append(priority_label)
                suggestions['reasoning'].append(f"Priority '{analysis['suggested_priority']}' based on content analysis")
            
            # Add type suggestions
            if analysis['is_bug'] and not any('bug' in label for label in analysis['current_labels']):
                suggestions['suggested_labels'].append('bug')
                suggestions['reasoning'].append("Appears to be a bug report")
                
            if analysis['is_feature'] and not any('enhancement' in label or 'feature' in label for label in analysis['current_labels']):
                suggestions['suggested_labels'].append('enhancement')
                suggestions['reasoning'].append("Appears to be a feature request")
            
            # Add special flags
            if analysis['needs_environment_info']:
                suggestions['suggested_labels'].append('needs-info')
                suggestions['reasoning'].append("Bug report missing environment information")
                
            return suggestions
            
        except requests.exceptions.RequestException as e:
            return {'error': f"Failed to fetch issue #{issue_number}: {e}"}


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="GitHub Issue Triager for napari",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python issue_triager.py --report                    # Generate full triage report
  python issue_triager.py --issue 1234                # Analyze specific issue
  python issue_triager.py --untriaged                 # Show untriaged issues
  python issue_triager.py --component qt              # Filter by component
  python issue_triager.py --export-json report.json  # Export report to JSON
        """
    )
    
    parser.add_argument('--token', 
                       help='GitHub personal access token (or set GITHUB_TOKEN env var)')
    parser.add_argument('--issue', type=int,
                       help='Analyze specific issue number')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive triage report')
    parser.add_argument('--untriaged', action='store_true',
                       help='Show issues that need triaging')
    parser.add_argument('--component',
                       help='Filter issues by component (qt, vispy, layers, etc.)')
    parser.add_argument('--since',
                       help='Fetch issues updated since date (ISO format)')
    parser.add_argument('--export-json',
                       help='Export report to JSON file')
    parser.add_argument('--max-issues', type=int, default=500,
                       help='Maximum number of issues to analyze (default: 500)')
    
    args = parser.parse_args()
    
    # Initialize triager
    triager = IssueTriage(token=args.token)
    
    if args.issue:
        # Analyze specific issue
        print(f"Analyzing issue #{args.issue}...")
        result = triager.suggest_labels_for_issue(args.issue)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return 1
            
        print(f"\nIssue #{result['issue_number']}: {result['title']}")
        print(f"Current labels: {', '.join(result['current_labels']) if result['current_labels'] else 'None'}")
        print(f"Suggested labels: {', '.join(result['suggested_labels'])}")
        print("\nReasoning:")
        for reason in result['reasoning']:
            print(f"  - {reason}")
            
    elif args.report or args.untriaged:
        # Generate report
        print("Fetching issues from napari/napari...")
        max_pages = max(1, args.max_issues // 100)
        issues = triager.fetch_issues(max_pages=max_pages, since=args.since)
        
        if not issues:
            print("No issues found.")
            return 0
            
        print(f"Analyzing {len(issues)} issues...")
        report = triager.generate_triage_report(issues)
        
        if args.export_json:
            with open(args.export_json, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report exported to {args.export_json}")
        
        # Display report
        print(f"\n=== Napari Issue Triage Report ===")
        print(f"Generated: {report['generated_at']}")
        print(f"Total issues analyzed: {report['total_issues']}")
        print(f"Issues needing triage: {report['needs_triage']}")
        print(f"Bug reports missing env info: {report['needs_env_info']}")
        
        if args.untriaged:
            print(f"\n--- Issues Needing Triage ---")
            untriaged_count = 0
            for issue in issues:
                analysis = triager.analyze_issue_content(issue)
                if analysis['needs_triage']:
                    print(f"#{issue['number']}: {issue['title'][:80]}...")
                    if analysis['suggested_components']:
                        print(f"  Suggested components: {', '.join(analysis['suggested_components'])}")
                    if analysis['suggested_priority']:
                        print(f"  Suggested priority: {analysis['suggested_priority']}")
                    untriaged_count += 1
                    if untriaged_count >= 10:  # Limit output
                        remaining = report['needs_triage'] - 10
                        if remaining > 0:
                            print(f"  ... and {remaining} more")
                        break
        else:
            # Show summary statistics
            print(f"\n--- Component Breakdown ---")
            for component, count in sorted(report['component_breakdown'].items(), key=lambda x: x[1], reverse=True):
                print(f"{component}: {count}")
                
            print(f"\n--- Priority Breakdown ---")
            for priority, count in sorted(report['priority_breakdown'].items(), key=lambda x: x[1], reverse=True):
                print(f"{priority}: {count}")
                
            print(f"\n--- Type Breakdown ---")
            for issue_type, count in sorted(report['type_breakdown'].items(), key=lambda x: x[1], reverse=True):
                print(f"{issue_type}: {count}")
        
        # Show recommendations
        if report['recommendations']:
            print(f"\n--- Recommendations ---")
            for rec in report['recommendations']:
                print(f"[{rec['type'].upper()}] {rec['message']}")
                if 'action' in rec:
                    print(f"  Action: {rec['action']}")
                if 'issues' in rec:
                    print("  Issues:")
                    for issue_ref in rec['issues']:
                        print(f"    - #{issue_ref['number']}: {issue_ref['title'][:60]}...")
                        
    else:
        parser.print_help()
        return 1
        
    return 0


if __name__ == '__main__':
    sys.exit(main())