"""
Custom exceptions for the GLB importer module.
"""


class ImporterError(Exception):
    """Base exception for importer errors."""
    pass


class GLBParseError(ImporterError):
    """Raised when GLB file cannot be parsed."""
    pass


class SkeletonError(ImporterError):
    """Raised when skeleton extraction fails."""
    pass


class MeshError(ImporterError):
    """Raised when mesh extraction fails."""
    pass


class AnimationError(ImporterError):
    """Raised when animation extraction fails."""
    pass


class CoordinateTransformError(ImporterError):
    """Raised when coordinate transformation fails."""
    pass


class MuJoCoGenerationError(ImporterError):
    """Raised when MuJoCo XML generation fails."""
    pass
