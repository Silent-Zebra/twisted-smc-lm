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

# Expose the main function for creating twists based on reward models
from .factory import TwistBuilderFactory as create_twist_builder

__all__ = [
    'TwistBuilder',
    'TwistConfig',
    'TwistBuilderFactory',
    'create_twist_builder',
] 