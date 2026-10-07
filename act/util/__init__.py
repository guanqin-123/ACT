#===- act/util/__init__.py - ACT Utility Package ---------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Utility package containing common functions for ACT network manipulation,
#   analysis, validation, and serialization utilities.
#
#===---------------------------------------------------------------------===#

# Device management utilities - import explicitly when needed to avoid triggering argparse
# from .device_manager import *
# Note: device_manager auto-initializes from command line args, so it should only be
# imported by modules that actually run as CLI tools, not by library code.

__all__ = [
    # Device management utilities should be imported explicitly:
    # from act.util.device_manager import get_default_device, get_default_dtype
]