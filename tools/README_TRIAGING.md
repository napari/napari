# Napari GitHub Issue Triaging System

This directory contains tools for triaging GitHub issues in the napari repository. The system helps maintainers and contributors efficiently categorize, prioritize, and manage issues.

## Components

### 1. `issue_triager.py` - Core Triaging Engine
The main triaging logic that analyzes issue content and suggests appropriate labels and actions.

**Features:**
- Content analysis using keyword matching
- Component detection (Qt, layers, vispy, etc.)
- Priority assessment (critical, high, medium, low)
- Bug vs feature request classification
- Missing information detection

**Usage:**
```bash
# Analyze a specific issue
python tools/issue_triager.py --issue 1234

# Generate comprehensive report
python tools/issue_triager.py --report

# Show untriaged issues
python tools/issue_triager.py --untriaged

# Export report to JSON
python tools/issue_triager.py --report --export-json triage_report.json
```

### 2. `enhanced_triager.py` - MCP Integration Layer
Enhanced triager that works with GitHub MCP server tools for real-time API access.

**Features:**
- Integration with GitHub MCP tools
- Batch label suggestions
- Maintainer action items
- Export capabilities

### 3. `triage_workflow.py` - Complete Workflow Demo
Demonstrates the complete issue triaging workflow with realistic examples.

**Features:**
- End-to-end triaging demonstration
- Label suggestion workflow
- Maintainer dashboard generation
- Action item prioritization

### 4. `live_triager.py` - Live Integration Demo
Shows how to integrate with live GitHub data using MCP tools.

**Features:**
- Real-time issue fetching simulation
- Dashboard export
- Quick action generation
- Integration point examples

## Issue Analysis

The triaging system analyzes issues based on:

### Component Detection
- **Qt**: GUI, widgets, dialogs, Qt-related keywords
- **VisPy**: OpenGL, rendering, canvas, shaders
- **Layers**: Image, labels, points, shapes, etc.
- **I/O**: File reading/writing, format support
- **Plugins**: Plugin system, npe2, extensions
- **Performance**: Speed, memory, optimization
- **Tests**: Testing, CI/CD, pytest
- **Installation**: Setup, dependencies, pip/conda
- **Preferences**: Settings, configuration

### Priority Assessment
- **Critical**: Crashes, data loss, urgent issues
- **High**: Major problems, regressions, blockers  
- **Medium**: Enhancements, improvements
- **Low**: Minor issues, cleanup, documentation

### Type Classification
- **Bug**: Error reports, crashes, unexpected behavior
- **Feature**: Enhancement requests, new functionality
- **Task**: Maintenance, refactoring, technical debt

## Usage Examples

### Basic Triaging
```python
from tools.issue_triager import IssueTriage

triager = IssueTriage()
analysis = triager.suggest_labels_for_issue(1234)
print(f"Suggested labels: {analysis['suggested_labels']}")
```

### Batch Analysis
```python
from tools.enhanced_triager import NapariIssueTriager

triager = NapariIssueTriager()
# Assume issues_data comes from GitHub MCP tools
report = triager.analyze_issues_from_api_response(issues_data)
suggestions = triager.generate_label_suggestions_batch(issues_data['issues'])
```

### Live Integration
```python
# This would be the actual integration pattern:
import github_mcp_server

# Fetch open issues
issues_response = github_mcp_server.list_issues(
    owner="napari", 
    repo="napari", 
    state="OPEN",
    perPage=50
)

# Analyze with triager
triager = NapariIssueTriager()
report = triager.analyze_issues_from_api_response(issues_response)

# Generate summary
summary = triager.generate_triage_summary(report)
print(summary)
```

## Integration Opportunities

### GitHub Actions
Create workflows that automatically:
- Suggest labels for new issues
- Request missing environment information
- Notify maintainers of critical issues
- Generate weekly triage reports

### Bot Integration
- Slack/Discord notifications for urgent issues
- Automated responses for common issue types
- Weekly digest for maintainers
- Integration with project management tools

### Dashboard
- Web interface for visual issue management
- Filtering and sorting by analysis results
- Bulk labeling interface
- Progress tracking

## Customization

The triaging system can be customized by modifying:

1. **Keyword mappings** in `IssueTriage.COMPONENT_KEYWORDS`
2. **Priority detection** in `IssueTriage.PRIORITY_KEYWORDS`
3. **Label suggestions** in the analysis methods
4. **Report formatting** in the summary generation

## Dependencies

- `requests` - For GitHub API access (when not using MCP tools)
- Python 3.10+ - For modern type hints and features

## Contributing

To improve the triaging system:

1. Add new component keywords for better detection
2. Enhance priority classification logic
3. Improve confidence scoring algorithms
4. Add support for more issue types
5. Create additional output formats

## Example Output

```
=== Napari Issue Triage Report ===
Generated: 2025-08-31T02:48:10.977532
Total issues analyzed: 4

--- Summary Statistics ---
Issues needing triage: 1
Bug reports missing env info: 3

--- Component Breakdown ---
installation: 2
layers: 2
qt: 1
performance: 1

--- Recommendations ---
[INFO_NEEDED] 3 bug reports are missing environment information
  Action: Ask reporters to run 'napari --info' and provide output

--- Issues Needing Attention ---
#8234: Qt bindings not found with uv tool commands...
  • Add component labels: qt, installation
  • Request environment information
```

This triaging system helps napari maintainers efficiently manage the large volume of issues by providing automated analysis, suggestions, and prioritization.