"""Noelle Desktop AI Studio.

Importing this package must stay cheap: the desktop shell imports it, and the
shell is not allowed to pull in torch, transformers or diffusers. Heavy imports
belong inside :mod:`noelle.runtime` functions that only run in worker processes.
"""

__version__ = "1.1.1"
APP_NAME = "Noelle"
APP_ID = "ai.noelle.studio"
