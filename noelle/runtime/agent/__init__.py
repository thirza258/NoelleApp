"""Agent Runtime: repository inspection and environment planning.

Pure Python and safe to import from the desktop shell; the actual venv work goes
through :mod:`noelle.runtime.environment_manager`.
"""

from .environment_planner import EnvironmentPlan, plan_for_verdict
from .repository_analyzer import ModelInspectionReport, RepositoryAnalyzer, analyze_local

__all__ = ["EnvironmentPlan", "ModelInspectionReport", "RepositoryAnalyzer", "analyze_local", "plan_for_verdict"]