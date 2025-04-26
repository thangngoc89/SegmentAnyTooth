# SPDX-License-Identifier: MIT
# ============================================================================
# SegmentAnyTooth
#
# Copyright (c) 2025 Khoa D. Nguyen
#
# This file is part of SegmentAnyTooth and is licensed under the MIT License.
# See LICENSE file in the repository root for full license information.
#
# Note: Pretrained model weights provided separately are under a Non-Commercial License.
# Refer to the WEIGHTS_LICENSE.txt for terms and conditions regarding model usage.
# ============================================================================
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """A context manager to suppress stdout temporarily."""
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout