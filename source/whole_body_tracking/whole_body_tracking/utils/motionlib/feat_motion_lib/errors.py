class MotionLibError(Exception):
    """Base exception for feat_motion_lib."""


class MotionFileError(MotionLibError):
    """Raised when a motion file cannot be loaded or validated."""


class MotionValidationError(MotionLibError):
    """Raised when motion data does not satisfy expected shapes or metadata."""


class MotionAlignmentError(MotionLibError):
    """Raised when joint/body names cannot be aligned."""


class MotionNotLoadedError(MotionLibError):
    """Raised when a query is made before loading motions."""
