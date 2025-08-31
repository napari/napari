#!/usr/bin/env python3
"""
Live Napari Issue Triaging with Real GitHub Data

This script uses the actual GitHub MCP server tools to demonstrate
live issue triaging on the napari repository.
"""

import json
import sys
from pathlib import Path

# Add tools directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_triager import NapariIssueTriager


def analyze_real_napari_issues():
    """Analyze real napari issues using the triaging system."""
    
    print("🔍 Live Analysis of Real Napari Issues")
    print("=" * 45)
    print()
    
    # Real GitHub API response from napari/napari (latest 10 open issues)
    real_github_data = {
        "issues": [
            {
                "number": 8234,
                "state": "OPEN",
                "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
                "body": "\n\nCreating a temp environment to launch napari with Qt6 does not work (found through debugging a different setup for someone): `uvx --with \"napari[pyqt6]\" napari`\n\nUsing Qt5 works as expected: `uvx -w \"napari[pyqt5]\" napari`\n\nTrying to create a tool install will also not work with  Qt6: `uv tool install --with \"napari[pyqt6]\" napari`, so its generally an issue with uv tool approaches. I would really have no idea why because the environment _does_ install Qt6 libraries 🤷 \n\nError message:\n```bash\nqtpy.QtBindingsNotFoundError: No Qt bindings could be found\n```",
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
                "body": "### 🐛 Bug Report\n\nWhen using napari 0.6.3 or later in combination with numpy 2.2.6, created shapes will show some strange behavior:\n- shapes will not be drawn in the right location\n- selecting a shape and moving it will visually affect another shape, possibly on another frame of the image\n\nnapari: 0.6.4\nPlatform: Linux-6.16.2-arch1-1-x86_64-with-glibc2.42\nPython: 3.13.7\nQt: 6.9.0\nNumPy: 2.2.6",
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
                "number": 8224,
                "state": "OPEN",
                "title": "Start using `array-api-strict` in our test suite", 
                "body": "## 🧰 Task\n\nIt will be nice to use [`array-api-strict`](https://pypi.org/project/array-api-strict/) in out base test suite to ensure wider napari compatibility with as many array backend as possible.",
                "user": {"login": "Czaki"},
                "labels": [{"name": "task", "description": "Tasks for contributors and maintainers"}],
                "comments": 0,
                "created_at": "2025-08-20T08:54:27Z",
                "updated_at": "2025-08-20T08:54:27Z"
            },
            {
                "number": 8220,
                "state": "OPEN",
                "title": "[test-bot] pip install --pre is failing",
                "body": "The --pre Test workflow failed on 2025-08-30 13:09 UTC\n\nThe most recent failing test was on macos-latest py3.12 pyqt6\nwith commit: 4b560f81e253c9fa1f1f7d9d7bfe3e4a15d07525",
                "user": {"login": "github-actions"},
                "labels": [{"name": "bug", "description": "Something isn't working as expected"}],
                "comments": 1,
                "created_at": "2025-08-17T01:04:16Z",
                "updated_at": "2025-08-30T13:09:09Z"
            },
            {
                "number": 8219,
                "state": "OPEN",
                "title": "Fix flaky Mac tests",
                "body": "test_toggle_fullscreen_from_maximized and test_screenshot. Maybe we should add the qtbot waits and qapp process events to these? These are likely the new test priorities to fix.",
                "user": {"login": "TimMonko"},
                "labels": [
                    {"name": "priority:high", "description": "High priority issue"},
                    {"name": "task", "description": "Tasks for contributors and maintainers"}
                ],
                "comments": 0,
                "created_at": "2025-08-16T16:49:31Z",
                "updated_at": "2025-08-16T16:49:55Z"
            },
            {
                "number": 8214,
                "state": "OPEN",
                "title": "Command/plugin double registration errors occurring from `napari==0.6.3`.",
                "body": "### 🐛 Bug Report\n\nI'm experiencing several `ValueError` of command double registrations, with plugin system. This is a plugins related bug with command registration.",
                "user": {"login": "LucaMarconato"},
                "labels": [{"name": "bug", "description": "Something isn't working as expected"}],
                "comments": 2,
                "created_at": "2025-08-14T13:49:21Z",
                "updated_at": "2025-08-14T14:23:42Z"
            },
            {
                "number": 8201,
                "state": "OPEN",
                "title": "Upgrade the transpose button to have a popup and also permit mirroring an axis",
                "body": "## 🚀 Feature\n\nAdd a popup menu to the transpose button with additional features:\n- mirror an axis, e.g. `data[:,::-1]`\n- rotate\n\nMirroring is pretty useful, when data collected by one modality is in a different orientation.",
                "user": {"login": "psobolewskiPhD"},
                "labels": [{"name": "feature", "description": "New feature or request"}],
                "comments": 0,
                "created_at": "2025-08-08T15:19:45Z",
                "updated_at": "2025-08-08T15:20:09Z"
            },
            {
                "number": 8197,
                "state": "OPEN",
                "title": "Richer import/export dialog to replace current complex opening workflow",
                "body": "",
                "user": {"login": "DragaDoncila"},
                "labels": [],
                "comments": 0,
                "created_at": "2025-08-06T04:14:56Z",
                "updated_at": "2025-08-06T04:14:56Z"
            },
            {
                "number": 8191,
                "state": "OPEN",
                "title": "Thick slicing followups",
                "body": "## 🧰 Task\n\n- Add chevrons to indicate that the dims arrow buttons -- and slider pill?? -- have a right click menu.\n- Add tooltip to the `projection mode` combobox in the Layer controls",
                "user": {"login": "psobolewskiPhD"},
                "labels": [{"name": "task", "description": "Tasks for contributors and maintainers"}],
                "comments": 0,
                "created_at": "2025-08-01T16:40:50Z",
                "updated_at": "2025-08-01T16:40:50Z"
            }
        ],
        "pageInfo": {
            "endCursor": "Y3Vyc29yOnYyOpK5MjAyNS0wOC0wMVQxMTo0MDo1MC0wNTowMM7DxbnW",
            "hasNextPage": True,
            "hasPreviousPage": False,
            "startCursor": "Y3Vyc29yOnYyOpK5MjAyNS0wOC0yOFQxNTo0NDowNC0wNTowMM7IjBO0"
        },
        "totalCount": 1029
    }
    
    # Initialize triager
    triager = NapariIssueTriager()
    
    print("📊 Analyzing 10 real open issues from napari/napari...")
    print()
    
    # Analyze the real issues
    report = triager.analyze_issues_from_api_response(real_github_data)
    
    # Generate and display the triage summary
    summary = triager.generate_triage_summary(report, 'text')
    print(summary)
    
    print("🏷️ DETAILED LABEL ANALYSIS")
    print("=" * 30)
    
    # Show detailed analysis for each issue
    label_suggestions = triager.generate_label_suggestions_batch(real_github_data['issues'])
    
    for issue_data in real_github_data['issues']:
        issue_num = issue_data['number']
        title = issue_data['title']
        current_labels = [label['name'] for label in issue_data.get('labels', [])]
        
        print(f"\n#{issue_num}: {title}")
        print(f"Current: {current_labels}")
        
        if issue_num in label_suggestions:
            print(f"Suggest: +{label_suggestions[issue_num]}")
        else:
            print("Suggest: Well labeled ✅")
        
        # Add analysis details
        issue_converted = triager.convert_github_issue_to_triager_format(issue_data)
        analysis = triager.triager.analyze_issue_content(issue_converted)
        
        details = []
        if analysis['suggested_priority']:
            details.append(f"Priority: {analysis['suggested_priority']}")
        if analysis['is_bug']:
            details.append("Type: Bug")
        if analysis['is_feature']:
            details.append("Type: Feature")
        if analysis['needs_environment_info']:
            details.append("⚠️ Missing env info")
        if analysis['needs_triage']:
            details.append("🔍 Needs triage")
            
        if details:
            print(f"Notes: {' | '.join(details)}")
    
    # Export detailed report
    output_file = "/tmp/napari_live_triage_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Full report exported to: {output_file}")
    
    # Summary for maintainers
    print(f"\n📋 MAINTAINER SUMMARY")
    print("=" * 25)
    
    needs_attention = sum(1 for item in report.get('detailed_analysis', []) 
                         if item['suggested_actions'])
    
    total_suggestions = sum(len(suggestions) for suggestions in label_suggestions.values())
    
    print(f"📈 Issues analyzed: {report['total_issues']}")
    print(f"🎯 Need attention: {needs_attention}")
    print(f"🏷️ Label suggestions: {total_suggestions}")
    print(f"⚠️ Missing env info: {report['needs_env_info']}")
    print(f"🔍 Need triage: {report['needs_triage']}")
    
    # Show top recommendations
    if report.get('recommendations'):
        print(f"\n⭐ TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"{i}. [{rec['type'].upper()}] {rec['message']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(analyze_real_napari_issues())