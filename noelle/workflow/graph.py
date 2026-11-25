"""Workflow Graph data model and validation for the Noelle Hybrid Lab.

Supports sequential and branching pipelines, cycle detection, edge binding,
and YAML/JSON serialization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from noelle.workflow.nodes import BaseNode, NODE_REGISTRY


class PrivacyMode(str, Enum):
    LOCAL_ONLY = "local"
    HYBRID = "hybrid"
    CLOUD_ONLY = "cloud"


@dataclass
class Edge:
    """Directed connection between an output port and an input port."""
    from_node: str
    from_port: str
    to_node: str
    to_port: str

    def to_dict(self) -> dict[str, str]:
        return {
            "from_node": self.from_node,
            "from_port": self.from_port,
            "to_node": self.to_node,
            "to_port": self.to_port,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Edge:
        return cls(
            from_node=data["from_node"],
            from_port=data["from_port"],
            to_node=data["to_node"],
            to_port=data["to_port"],
        )


@dataclass
class WorkflowGraph:
    """Directed acyclic graph representation of an AI workflow."""
    id: str = "workflow-default"
    name: str = "Untitled Workflow"
    description: str = ""
    privacy_mode: PrivacyMode = PrivacyMode.HYBRID
    nodes: dict[str, BaseNode] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: BaseNode) -> None:
        self.nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges if e.from_node != node_id and e.to_node != node_id]

    def connect(self, from_node: str, from_port: str, to_node: str, to_port: str) -> None:
        # Avoid duplicate edges
        for e in self.edges:
            if (e.from_node == from_node and e.from_port == from_port
                    and e.to_node == to_node and e.to_port == to_port):
                return
        self.edges.append(Edge(from_node, from_port, to_node, to_port))

    def disconnect(self, from_node: str, from_port: str, to_node: str, to_port: str) -> None:
        self.edges = [
            e for e in self.edges
            if not (e.from_node == from_node and e.from_port == from_port
                    and e.to_node == to_node and e.to_port == to_port)
        ]

    def has_cloud_nodes(self) -> bool:
        return any(n.is_cloud for n in self.nodes.values())

    def get_cloud_nodes(self) -> list[BaseNode]:
        return [n for n in self.nodes.values() if n.is_cloud]

    def topological_sort(self) -> list[str]:
        """Return node IDs in topologically sorted execution order.

        Raises:
            ValueError: If a directed cycle is detected.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}

        for edge in self.edges:
            if edge.from_node in self.nodes and edge.to_node in self.nodes:
                adj[edge.from_node].append(edge.to_node)
                in_degree[edge.to_node] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("Directed cycle detected in workflow graph")

        return order

    def validate(self) -> list[str]:
        """Validate integrity of node connections and provider configurations."""
        issues: list[str] = []
        for edge in self.edges:
            if edge.from_node not in self.nodes:
                issues.append(f"Edge references nonexistent source node '{edge.from_node}'")
            if edge.to_node not in self.nodes:
                issues.append(f"Edge references nonexistent destination node '{edge.to_node}'")

        try:
            self.topological_sort()
        except ValueError as exc:
            issues.append(str(exc))

        if self.privacy_mode == PrivacyMode.LOCAL_ONLY and self.has_cloud_nodes():
            cloud_names = [f"'{n.id}' ({n.provider_id})" for n in self.get_cloud_nodes()]
            issues.append(f"Privacy Mode is Local-Only, but cloud nodes exist: {', '.join(cloud_names)}")

        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "privacy_mode": self.privacy_mode.value,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowGraph:
        graph = cls(
            id=data.get("id", "workflow-default"),
            name=data.get("name", "Untitled Workflow"),
            description=data.get("description", ""),
            privacy_mode=PrivacyMode(data.get("privacy_mode", "hybrid")),
        )
        for nd in data.get("nodes", []):
            ntype = nd.get("node_type", "base")
            node_cls = NODE_REGISTRY.get(ntype, BaseNode)
            node = node_cls.from_dict(nd)
            graph.add_node(node)
        for ed in data.get("edges", []):
            graph.edges.append(Edge.from_dict(ed))
        return graph

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> WorkflowGraph:
        return cls.from_dict(json.loads(json_str))
