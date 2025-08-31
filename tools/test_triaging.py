#!/usr/bin/env python3
"""
Tests for the napari issue triaging system.
"""

import sys
from pathlib import Path

# Add tools directory to path
sys.path.append(str(Path(__file__).parent))

from issue_triager import IssueTriage


def test_component_detection():
    """Test component detection functionality."""
    triager = IssueTriage()
    
    # Test Qt component detection
    qt_issue = {
        'title': 'Widget dialog crashes when clicking button',
        'body': 'The Qt widget dialog crashes with a segfault when I click the OK button.',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(qt_issue)
    assert 'qt' in analysis['suggested_components'], "Should detect Qt component"
    assert analysis['is_bug'], "Should detect as bug"
    
    # Test layers component detection
    layers_issue = {
        'title': 'Image layer not displaying correctly',
        'body': 'The image layer shows incorrect colors and shapes are not visible.',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(layers_issue)
    assert 'layers' in analysis['suggested_components'], "Should detect layers component"
    
    # Test vispy component detection
    vispy_issue = {
        'title': 'OpenGL rendering error with shaders',
        'body': 'Canvas rendering fails with OpenGL shader compilation errors.',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(vispy_issue)
    assert 'vispy' in analysis['suggested_components'], "Should detect vispy component"
    
    print("✅ Component detection tests passed")


def test_priority_detection():
    """Test priority detection functionality."""
    triager = IssueTriage()
    
    # Test critical priority
    critical_issue = {
        'title': 'Critical crash causing data loss',
        'body': 'Napari crashes and user loses all their work. This is urgent.',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(critical_issue)
    assert analysis['suggested_priority'] == 'critical', "Should detect critical priority"
    
    # Test enhancement priority
    feature_issue = {
        'title': 'Enhancement: Add new visualization feature',
        'body': 'It would be nice to add support for new visualization types.',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(feature_issue)
    assert analysis['suggested_priority'] == 'medium', "Should detect medium priority for enhancements"
    
    print("✅ Priority detection tests passed")


def test_bug_vs_feature_detection():
    """Test bug vs feature request detection."""
    triager = IssueTriage()
    
    # Test bug detection
    bug_issue = {
        'title': 'Error when loading file',
        'body': 'Getting an exception when trying to load my image file. Traceback shows FileNotFoundError.',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(bug_issue)
    assert analysis['is_bug'], "Should detect as bug"
    assert not analysis['is_feature'], "Should not detect as feature"
    
    # Test feature detection
    feature_issue = {
        'title': 'Feature request: Add support for new format',
        'body': 'Would be nice to add support for XYZ file format. Can we implement this?',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(feature_issue)
    assert analysis['is_feature'], "Should detect as feature"
    
    print("✅ Bug vs feature detection tests passed")


def test_environment_info_detection():
    """Test detection of missing environment information."""
    triager = IssueTriage()
    
    # Bug without environment info
    bug_no_env = {
        'title': 'Napari crashes on startup',
        'body': 'Napari crashes when I try to start it. Error message appears.',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(bug_no_env)
    assert analysis['needs_environment_info'], "Should detect missing environment info"
    
    # Bug with environment info
    bug_with_env = {
        'title': 'Napari crashes on startup', 
        'body': 'Napari crashes when I try to start it.\n\nnapari: 0.4.18\nPlatform: macOS-13.2.1\nPython: 3.11.4',
        'labels': []
    }
    
    analysis = triager.analyze_issue_content(bug_with_env)
    assert not analysis['needs_environment_info'], "Should not flag when env info present"
    
    print("✅ Environment info detection tests passed")


def test_report_generation():
    """Test report generation functionality."""
    triager = IssueTriage()
    
    sample_issues = [
        {
            'title': 'Qt widget bug',
            'body': 'Widget crashes',
            'labels': [],
            'number': 1,
            'updated_at': '2025-08-31T00:00:00Z',
            'user': {'login': 'test_user'}
        },
        {
            'title': 'Feature request for layers',
            'body': 'Enhancement to add new layer type',
            'labels': [{'name': 'enhancement'}],
            'number': 2,
            'updated_at': '2025-08-31T00:00:00Z',
            'user': {'login': 'test_user'}
        }
    ]
    
    report = triager.generate_triage_report(sample_issues)
    
    assert report['total_issues'] == 2, "Should count all issues"
    assert 'qt' in report['component_breakdown'], "Should detect Qt component"
    assert 'layers' in report['component_breakdown'], "Should detect layers component"
    assert report['type_breakdown']['bug'] >= 1, "Should count bugs"
    assert report['type_breakdown']['feature'] >= 1, "Should count features"
    
    print("✅ Report generation tests passed")


def run_all_tests():
    """Run all triaging tests."""
    print("🧪 Running Napari Issue Triaging Tests")
    print("=" * 40)
    
    try:
        test_component_detection()
        test_priority_detection()
        test_bug_vs_feature_detection()
        test_environment_info_detection()
        test_report_generation()
        
        print("\n🎉 All tests passed!")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())