"""Workflow execution runner for the Noelle Hybrid Lab.

Executes topologically ordered processing pipelines with boundary privacy enforcement,
sequential local GPU safety, and comprehensive telemetry aggregation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from noelle.providers.base import TelemetryRecord
from noelle.workflow.graph import PrivacyMode, WorkflowGraph

log = logging.getLogger("noelle.workflow.runner")


class PrivacyViolationError(RuntimeError):
    """Raised when an operation attempts external network egress in Local-Only mode."""
    pass


@dataclass
class WorkflowResult:
    """Consolidated result returned after executing a workflow graph."""
    workflow_id: str
    workflow_name: str
    is_success: bool
    total_latency_ms: float
    total_cost_usd: float
    total_tokens: int
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    telemetry: dict[str, TelemetryRecord] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "is_success": self.is_success,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "node_outputs": {
                k: {
                    pk: (pv if not isinstance(pv, bytes) else f"<binary {len(pv)} bytes>")
                    for pk, pv in v.items() if pk != "response"
                }
                for k, v in self.node_outputs.items()
            },
            "telemetry": {k: v.to_dict() for k, v in self.telemetry.items()},
            "error_message": self.error_message,
        }


class WorkflowExecutionContext:
    """State and telemetry collector passed through node execution."""

    def __init__(self, privacy_mode: PrivacyMode = PrivacyMode.HYBRID) -> None:
        self.privacy_mode = privacy_mode
        self.node_outputs: dict[str, dict[str, Any]] = {}
        self.telemetry_records: dict[str, TelemetryRecord] = {}

    def record_telemetry(self, node_id: str, record: TelemetryRecord) -> None:
        self.telemetry_records[node_id] = record

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.telemetry_records.values())

    @property
    def total_tokens(self) -> int:
        return sum(t.input_tokens + t.output_tokens for t in self.telemetry_records.values())


class WorkflowRunner:
    """Executes a WorkflowGraph with privacy checks and edge value routing."""

    def __init__(self, node_callback: Callable[[str, str], None] | None = None) -> None:
        self.node_callback = node_callback

    def execute(self, graph: WorkflowGraph, initial_inputs: dict[str, Any] | None = None,
                consent_given: bool = False) -> WorkflowResult:
        """Run workflow graph to completion.

        Consent defaults to **false**: a graph with cloud nodes fails closed unless the caller
        (UI/CLI) explicitly recorded the user's consent for this run.
        """
        start_time = time.perf_counter()

        # 1. Privacy Governance Check
        if graph.privacy_mode == PrivacyMode.LOCAL_ONLY and graph.has_cloud_nodes():
            cloud_nodes = [f"'{n.id}' ({n.provider_id})" for n in graph.get_cloud_nodes()]
            err_msg = (
                f"Zero Egress Violation: Privacy mode is Local-Only, but workflow contains "
                f"cloud nodes: {', '.join(cloud_nodes)}"
            )
            log.error(err_msg)
            raise PrivacyViolationError(err_msg)

        if graph.privacy_mode == PrivacyMode.HYBRID and graph.has_cloud_nodes() and not consent_given:
            err_msg = "Hybrid Privacy Mode requires explicit user consent before transmitting data to cloud endpoints."
            log.warning(err_msg)
            raise PermissionError(err_msg)

        # 2. Validation & Ordering
        try:
            exec_order = graph.topological_sort()
        except Exception as exc:
            return WorkflowResult(
                workflow_id=graph.id,
                workflow_name=graph.name,
                is_success=False,
                total_latency_ms=0.0,
                total_cost_usd=0.0,
                total_tokens=0,
                error_message=f"Invalid graph topology: {exc}",
            )

        ctx = WorkflowExecutionContext(graph.privacy_mode)
        if initial_inputs:
            for k, v in initial_inputs.items():
                ctx.node_outputs[k] = {"output": v, "value": v}

        # 3. Step-by-step Topological Execution
        for node_id in exec_order:
            node = graph.nodes[node_id]
            if self.node_callback:
                self.node_callback(node_id, "running")

            # Resolve input values from incoming edges
            resolved_inputs: dict[str, Any] = {}
            for edge in graph.edges:
                if edge.to_node == node_id:
                    src_outputs = ctx.node_outputs.get(edge.from_node, {})
                    if edge.from_port in src_outputs:
                        resolved_inputs[edge.to_port] = src_outputs[edge.from_port]
                    elif "output" in src_outputs:
                        resolved_inputs[edge.to_port] = src_outputs["output"]

            try:
                outputs = node.execute(resolved_inputs, ctx)
                ctx.node_outputs[node_id] = outputs
                if self.node_callback:
                    self.node_callback(node_id, "done")
            except Exception as exc:
                latency = (time.perf_counter() - start_time) * 1000.0
                log.exception("Node execution error at '%s': %s", node_id, exc)
                if self.node_callback:
                    self.node_callback(node_id, "error")
                return WorkflowResult(
                    workflow_id=graph.id,
                    workflow_name=graph.name,
                    is_success=False,
                    total_latency_ms=latency,
                    total_cost_usd=ctx.total_cost_usd,
                    total_tokens=ctx.total_tokens,
                    node_outputs=ctx.node_outputs,
                    telemetry=ctx.telemetry_records,
                    error_message=f"Node '{node_id}' failed: {exc}",
                )

        total_latency = (time.perf_counter() - start_time) * 1000.0
        return WorkflowResult(
            workflow_id=graph.id,
            workflow_name=graph.name,
            is_success=True,
            total_latency_ms=total_latency,
            total_cost_usd=ctx.total_cost_usd,
            total_tokens=ctx.total_tokens,
            node_outputs=ctx.node_outputs,
            telemetry=ctx.telemetry_records,
        )
