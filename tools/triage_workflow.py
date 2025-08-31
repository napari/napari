#!/usr/bin/env python3
"""
Complete Napari Issue Triaging Workflow

This script demonstrates how to use the GitHub MCP server tools with the
napari issue triager to perform comprehensive issue management.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add tools directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_triager import NapariIssueTriager
except ImportError:
    print("Error: Could not import enhanced_triager. Make sure it's in the same directory.")
    sys.exit(1)


def demonstrate_issue_triaging():
    """Demonstrate the complete issue triaging workflow."""
    
    print("🔍 Napari Issue Triaging Workflow Demonstration")
    print("=" * 55)
    print()
    
    # Initialize the triager
    triager = NapariIssueTriager()
    
    # Simulate analysis using recent napari issues
    # In a real scenario, this would use the GitHub MCP tools to fetch live data
    sample_response = {
        "issues": [
            {
                "number": 8234,
                "state": "OPEN",
                "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
                "body": "Creating a temp environment to launch napari with Qt6 does not work (found through debugging a different setup for someone): `uvx --with \"napari[pyqt6]\" napari`\n\nUsing Qt5 works as expected: `uvx -w \"napari[pyqt5]\" napari`\n\nTrying to create a tool install will also not work with Qt6: `uv tool install --with \"napari[pyqt6]\" napari`, so its generally an issue with uv tool approaches. Error message:\n```bash\nTraceback (most recent call last):\n  File \"qtpy\\__init__.py\", line 293, in <module>\n    raise QtBindingsNotFoundError from None\nqtpy.QtBindingsNotFoundError: No Qt bindings could be found\n```",
                "user": {"login": "TimMonko"},
                "labels": [{"name": "bug", "description": "Something isn't working as expected"}],
                "comments": 3,
                "created_at": "2025-08-28T20:44:04Z",
                "updated_at": "2025-08-28T21:00:17Z"
            },
            {
                "number": 8230,
                "state": "OPEN",
                "title": "bugs with stacked shapes layer since 0.6.3",
                "body": "### 🐛 Bug Report\n\nWhen using napari 0.6.3 or later in combination with numpy 2.2.6, created shapes will show some strange behavior:\n- shapes will not be drawn in the right location\n- selecting a shape and moving it will visually affect another shape, possibly on another frame of the image\n- selecting a shape after it has been moved like this will not work if clicked\n\nnapari: 0.6.4\nPlatform: Linux-6.16.2-arch1-1-x86_64-with-glibc2.42\nSystem: Arch Linux\nPython: 3.13.7\nQt: 6.9.0\nNumPy: 2.2.6",
                "user": {"login": "thopp-tudelft"},
                "labels": [{"name": "bug", "description": "Something isn't working as expected"}],
                "comments": 4,
                "created_at": "2025-08-25T12:25:47Z",
                "updated_at": "2025-08-26T05:41:04Z"
            },
            {
                "number": 8225,
                "state": "OPEN",
                "title": "Consider using `array-api-compat` as comptybility backend for different array backends",
                "body": "## 🧰 Task\n\nCheck if we could use [`array-api-compat`](https://pypi.org/project/array-api-compat/) as a compat layer to reduce code required to handle different array backends.",
                "user": {"login": "Czaki"},
                "labels": [{"name": "task", "description": "Tasks for contributors and maintainers"}],
                "comments": 3,
                "created_at": "2025-08-20T09:00:06Z",
                "updated_at": "2025-08-22T13:54:44Z"
            },
            {
                "number": 8220,
                "state": "OPEN",
                "title": "[test-bot] pip install --pre is failing",
                "body": "The --pre Test workflow failed on 2025-08-30 13:09 UTC\n\nThe most recent failing test was on macos-latest py3.12 pyqt6\nwith commit: 4b560f81e253c9fa1f1f7d9d7bfe3e4a15d07525\n\nFull run: https://github.com/napari/napari/actions/runs/17343543131",
                "user": {"login": "github-actions"},
                "labels": [{"name": "bug", "description": "Something isn't working as expected"}],
                "comments": 1,
                "created_at": "2025-08-17T01:04:16Z",
                "updated_at": "2025-08-30T13:09:09Z"
            },
            {
                "number": 8210,
                "state": "OPEN",
                "title": "Feature request: Add support for zarr v3",
                "body": "## ✨ Feature Request\n\nWould be great to add support for zarr v3 format for large image datasets. This would improve performance for multi-dimensional data.\n\n### Proposed solution\nIntegrate zarr v3 reading capabilities into the io layer.\n\n### Additional context\nZarr v3 has improved chunking and compression features.",
                "user": {"login": "contributor123"},
                "labels": [],
                "comments": 0,
                "created_at": "2025-08-15T10:30:00Z",
                "updated_at": "2025-08-15T10:30:00Z"
            }
        ],
        "pageInfo": {"hasNextPage": True, "totalCount": 1029}
    }
    
    # Analyze the issues
    print("1. Fetching and analyzing issues...")
    report = triager.analyze_issues_from_api_response(sample_response)
    
    print("2. Generating triage report...")
    summary = triager.generate_triage_summary(report, 'text')
    print(summary)
    
    # Show specific label suggestions
    print("=" * 55)
    print("🏷️  LABEL SUGGESTIONS BY ISSUE")
    print("=" * 55)
    
    suggestions = triager.generate_label_suggestions_batch(sample_response['issues'])
    
    for issue_data in sample_response['issues']:
        issue_num = issue_data['number']
        title = issue_data['title']
        current_labels = [label['name'] for label in issue_data.get('labels', [])]
        
        print(f"\n#{issue_num}: {title[:70]}...")
        print(f"Current: {', '.join(current_labels) if current_labels else 'None'}")
        
        if issue_num in suggestions:
            print(f"Suggest: +{', '.join(suggestions[issue_num])}")
        else:
            print("Suggest: No additional labels needed")
    
    # Export detailed report
    output_file = "/tmp/napari_triage_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report exported to: {output_file}")
    
    # Summary for maintainers
    print("\n" + "=" * 55)
    print("📋 MAINTAINER ACTION ITEMS")
    print("=" * 55)
    
    action_items = []
    for item in report.get('detailed_analysis', []):
        issue = item['issue']
        actions = item['suggested_actions']
        if actions:
            action_items.append({
                'issue_number': issue['number'],
                'title': issue['title'],
                'actions': actions,
                'priority': item['analysis'].get('suggested_priority', 'normal')
            })
    
    # Sort by priority
    priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'normal': 4}
    action_items.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    for item in action_items:
        priority_indicator = "🔥" if item['priority'] in ['critical', 'high'] else "📌"
        print(f"\n{priority_indicator} #{item['issue_number']}: {item['title'][:60]}...")
        for action in item['actions']:
            print(f"   • {action}")
    
    if not action_items:
        print("\n✅ No immediate action items - all issues are properly triaged!")
    
    print(f"\n🎯 Summary: {len(action_items)} issues need attention")
    return 0


if __name__ == '__main__':
    sys.exit(demonstrate_issue_triaging())