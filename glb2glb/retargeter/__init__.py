"""
GLB2GLB Retargeter Module
========================

This module provides IK-based motion retargeting between different models.

Note: IK-based retargeting requires the mink library:
  pip install pink
"""

import warnings

# IK-based retargeting
try:
    from .ik_based_retargeting import (
        IKBasedRetargeter,
        ik_retarget_motion
    )
    _ik_available = True
except ImportError as e:
    _ik_available = False
    warnings.warn(f"IK-based retargeting not available (mink required): {e}")

# Visualization utilities
try:
    from .visualize_retargeting import (
        DualRobotViewer,
        compare_retargeting_results,
        create_test_motion
    )
    _visualization_available = True
except ImportError as e:
    _visualization_available = False
    # Silently skip - visualization is optional

# Build __all__ based on available components
__all__ = []

if _ik_available:
    __all__.extend([
        'IKBasedRetargeter',
        'ik_retarget_motion',
    ])

if _visualization_available:
    __all__.extend([
        'DualRobotViewer',
        'compare_retargeting_results',
        'create_test_motion',
    ])

# Helper function to check available features
def check_retargeter_dependencies():
    """Check and report the status of retargeter dependencies."""
    print("GLB2GLB Retargeter Module Status:")
    print("-" * 40)
    print(f"✓ IK retargeting: {'Available' if _ik_available else 'Not available (install mink)'}")
    print(f"✓ Visualization: {'Available' if _visualization_available else 'Not available'}")
    
    if not _ik_available:
        print("\nTo install mink for IK-based retargeting:")
        print("  pip install pink")
    
    return {
        'ik': _ik_available,
        'visualization': _visualization_available
    }

__all__.append('check_retargeter_dependencies')