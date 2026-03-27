# Device management utilities - import explicitly when needed to avoid triggering argparse
# from .device_manager import *
# Note: device_manager auto-initializes from command line args, so it should only be
# imported by modules that actually run as CLI tools, not by library code.

__all__ = [
    # Device management utilities should be imported explicitly:
    # from act.util.device_manager import get_default_device, get_default_dtype
]