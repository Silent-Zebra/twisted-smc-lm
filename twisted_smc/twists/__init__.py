"""Twist implementations for modifying language model distributions.

This package provides a modular framework for defining and applying twists to
language model distributions. Twists are functions that modify the base model's
output distribution according to various criteria like toxicity, sentiment,
or specific continuations.

The package implements the Factory and Builder design patterns to support:
1. Creating the appropriate twist builders based on reward model types
2. Building twist functions with proper configuration
3. Collecting true posterior samples where possible
"""

from .base import TwistBuilder, TwistConfig
from .factory import TwistBuilderFactory
from .api import get_log_true_final_twists


__all__ = [
    'TwistBuilder',
    'TwistConfig',
    'TwistBuilderFactory',
    'get_log_true_final_twists'
] 