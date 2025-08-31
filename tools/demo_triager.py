#!/usr/bin/env python3
"""
Demo script to showcase the issue triaging functionality using the GitHub MCP tools.
"""

from issue_triager import IssueTriage

def demo_issue_analysis():
    """Demonstrate issue analysis on sample napari issues."""
    
    # Sample issues from napari repository (extracted from API response)
    sample_issues = [
        {
            "number": 8234,
            "title": "Qt bindings are (sometimes) not found with `uv` tool commands and `Qt6`",
            "body": "Creating a temp environment to launch napari with Qt6 does not work. Using Qt5 works as expected. Error message shows QtBindingsNotFoundError: No Qt bindings could be found.",
            "labels": [{"name": "bug"}],
            "updated_at": "2025-08-28T21:00:17Z",
            "user": {"login": "TimMonko"}
        },
        {
            "number": 8230,
            "title": "bugs with stacked shapes layer since 0.6.3",
            "body": "When using napari 0.6.3 or later in combination with numpy 2.2.6, created shapes will show some strange behavior: shapes will not be drawn in the right location, selecting a shape and moving it will visually affect another shape",
            "labels": [{"name": "bug"}],
            "updated_at": "2025-08-26T05:41:04Z",
            "user": {"login": "thopp-tudelft"}
        },
        {
            "number": 8225,
            "title": "Consider using `array-api-compat` as comptybility backend for different array backends",
            "body": "Check if we could use array-api-compat as a compat layer to reduce code required to handle different array backends.",
            "labels": [{"name": "task"}],
            "updated_at": "2025-08-22T13:54:44Z",
            "user": {"login": "Czaki"}
        },
        {
            "number": 8224,
            "title": "Start using `array-api-strict` in our test suite",
            "body": "It will be nice to use array-api-strict in our base test suite to ensure wider napari compatibility with as many array backend as possible.",
            "labels": [{"name": "task"}],
            "updated_at": "2025-08-20T08:54:27Z",
            "user": {"login": "Czaki"}
        },
        {
            "number": 8220,
            "title": "[test-bot] pip install --pre is failing",
            "body": "The --pre Test workflow failed on 2025-08-30 13:09 UTC. The most recent failing test was on macos-latest py3.12 pyqt6",
            "labels": [{"name": "bug"}],
            "updated_at": "2025-08-30T13:09:09Z",
            "user": {"login": "github-actions"}
        }
    ]
    
    # Initialize triager
    triager = IssueTriage()
    
    print("=== Napari Issue Triaging Demo ===\n")
    
    for issue in sample_issues:
        print(f"Analyzing Issue #{issue['number']}: {issue['title']}")
        print("-" * 60)
        
        # Analyze the issue
        analysis = triager.analyze_issue_content(issue)
        
        # Display current labels
        current_labels = [label['name'] for label in issue['labels']]
        print(f"Current labels: {', '.join(current_labels) if current_labels else 'None'}")
        
        # Display suggested components
        if analysis['suggested_components']:
            print(f"Suggested components: {', '.join(analysis['suggested_components'])}")
        
        # Display suggested priority
        if analysis['suggested_priority']:
            print(f"Suggested priority: {analysis['suggested_priority']}")
        
        # Display type classification
        types = []
        if analysis['is_bug']:
            types.append('bug')
        if analysis['is_feature']:
            types.append('feature request')
        if types:
            print(f"Issue type: {', '.join(types)}")
        
        # Display special flags
        if analysis['needs_triage']:
            print("🔍 NEEDS TRIAGE")
        if analysis['needs_environment_info']:
            print("ℹ️ NEEDS ENVIRONMENT INFO")
        
        # Show confidence scores for debugging
        if analysis['confidence_scores']:
            print("Confidence scores:", end=" ")
            scores = [f"{k}:{v}" for k, v in analysis['confidence_scores'].items() if v > 0]
            print(", ".join(scores))
        
        print()
    
    # Generate a mini report
    print("=== Summary Report ===")
    report = triager.generate_triage_report(sample_issues)
    
    print(f"Total issues analyzed: {report['total_issues']}")
    print(f"Issues needing triage: {report['needs_triage']}")
    print(f"Bug reports missing env info: {report['needs_env_info']}")
    
    if report['component_breakdown']:
        print("\nComponent breakdown:")
        for component, count in sorted(report['component_breakdown'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {component}: {count}")
    
    if report['priority_breakdown']:
        print("\nPriority breakdown:")
        for priority, count in sorted(report['priority_breakdown'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {priority}: {count}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  [{rec['type'].upper()}] {rec['message']}")


if __name__ == '__main__':
    demo_issue_analysis()