#!/usr/bin/env python3
"""
Entry point for VNNLIB CLI.

Allows running the VNNLIB-specific CLI via: python -m act.front_end.vnnlib

This provides VNNLIB-specific commands for managing VNN-COMP benchmark
categories, parsing VNNLIB files, and working with ONNX models.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from act.front_end.vnnlib.cli import main

if __name__ == "__main__":
    main()
