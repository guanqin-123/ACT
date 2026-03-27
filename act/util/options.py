    
import os
from act.util.path_config import get_pipeline_log_dir


class PerformanceOptions:
    """Global performance and debugging options.
    
    Both options default to True (debugging enabled) to help catch bugs during development.
    Use disable_all() for production/performance runs.
    
    Attributes:
        debug_tf: Enable debug logging in transfer functions (default: True)
        validate_constraints: Enable constraint validation (default: True)
        debug_output_file: Path to debug log file (default: "act/pipeline/log/act_debug_tf.log")
        debug_tf_max_constraints: Maximum number of constraints to log per layer (default: 50)
    
    Example:
        # Default: debugging enabled
        from act.util.options import PerformanceOptions
        
        # Performance mode
        PerformanceOptions.disable_all()
        
        # Custom debug file
        PerformanceOptions.set_debug_output_file("my_debug.log")
        
        # Show more/fewer constraints in debug log
        PerformanceOptions.debug_tf_max_constraints = 100
    """
    debug_tf: bool = True
    validate_constraints: bool = True
    debug_output_file: str = os.path.join(get_pipeline_log_dir(), "act_debug_tf.log")
    debug_tf_max_constraints: int = 50
    
    @classmethod
    def enable_debug_tf(cls) -> None:
        """Enable transfer function debug logging."""
        cls.debug_tf = True
    
    @classmethod
    def enable_validate_constraints(cls) -> None:
        """Enable constraint validation."""
        cls.validate_constraints = True
    
    @classmethod
    def disable_all(cls) -> None:
        """Disable all debugging features for performance runs."""
        cls.debug_tf = False
        cls.validate_constraints = False
    
    @classmethod
    def set_debug_output_file(cls, filepath: str) -> None:
        """Set the debug output file path.
        
        Args:
            filepath: Path to the debug output file
        """
        cls.debug_output_file = filepath