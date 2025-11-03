#!/usr/bin/env python3
"""
ACT Pipeline Entry Point.

Provides unified command-line interface for ACT pipeline operations including:
- ACTFuzzer: Inference-based whitebox fuzzing
- Model synthesis and inference (future)
- Verification workflows (future)

Usage:
    python -m act.pipeline --list
    python -m act.pipeline --fuzz --category acasxu_2023

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from act.pipeline.cli import main

if __name__ == "__main__":
    main()
