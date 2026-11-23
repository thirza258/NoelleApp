"""Out-of-process compute plane.

Only :mod:`noelle.runtime.protocol`, :mod:`noelle.runtime.preflight` and
:mod:`noelle.runtime.client` are safe to import from the desktop shell. Every
other module may import torch and must only be imported inside worker,
server or download child processes.
"""
