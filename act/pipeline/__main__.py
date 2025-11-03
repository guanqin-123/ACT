#!/usr/bin/env python3
"""
ACT Pipeline Entry Point.

Provides unified command-line interface for ACT pipeline operations including:
- ACTFuzzer: Inference-based whitebox fuzzing
- Model synthesis and inference (future)
- Verification workflows (future)

Usage:
    # List available VNNLib benchmarks
    python -m act.pipeline --list
    
    # Fuzz CIFAR-100 VNNLib benchmark
    python -m act.pipeline --fuzz --category cifar100_2024 --timeout 60
    
    # Fuzz with custom iterations
    python -m act.pipeline --fuzz --category acasxu_2023 --iterations 5000

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

# CRITICAL FIX: Must hide sys.argv BEFORE any imports to prevent device_manager
# from calling get_parser() which creates the verification CLI parser.
# This needs to happen at module level, not inside if __name__ == "__main__"
import sys
_original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]  # Hide all args during imports

# Now safe to import (device_manager won't see --help or other args)
from act.pipeline.cli import main as fuzzing_main

if __name__ == "__main__":
    # Restore original argv so the fuzzing CLI can parse it correctly
    sys.argv = _original_argv
    
    # Run the fuzzing CLI
    fuzzing_main()
