"""Cost Accounting and Cloud vs. Local Decision Advisor.

Provides accurate total-cost-of-ownership (TCO) calculations, break-even
analysis, and strategic architecture recommendations for scaling AI workloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from noelle.providers.pricing import calculate_cost


@dataclass
class LocalHardwareParams:
    """Hardware profile and utility parameters for local compute costing."""
    hardware_cost_usd: float = 2000.0       # e.g. dedicated GPU workstation
    amortization_months: int = 24           # 2-year amortization schedule
    power_draw_watts: float = 450.0         # average active wattage under inference load
    electricity_kwh_cost: float = 0.15      # electricity rate per kWh in USD
    active_hours_per_day: float = 8.0       # active daily operational hours

    @property
    def monthly_amortization_usd(self) -> float:
        return self.hardware_cost_usd / max(1, self.amortization_months)

    @property
    def monthly_electricity_usd(self) -> float:
        kwh_per_day = (self.power_draw_watts / 1000.0) * self.active_hours_per_day
        return kwh_per_day * 30.0 * self.electricity_kwh_cost

    @property
    def total_monthly_cost_usd(self) -> float:
        return self.monthly_amortization_usd + self.monthly_electricity_usd


@dataclass
class DecisionAssessment:
    """Actionable economic and architectural comparison."""
    projected_monthly_queries: int
    avg_input_tokens: int
    avg_output_tokens: int
    cloud_model: str
    cloud_monthly_cost_usd: float
    local_monthly_cost_usd: float
    monthly_savings_usd: float
    savings_percentage: float
    break_even_queries_per_day: int
    break_even_days: float
    recommended_strategy: str
    strategic_rationale: str
    latency_cloud_ms: float
    latency_local_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "projected_monthly_queries": self.projected_monthly_queries,
            "cloud_model": self.cloud_model,
            "cloud_monthly_cost_usd": round(self.cloud_monthly_cost_usd, 2),
            "local_monthly_cost_usd": round(self.local_monthly_cost_usd, 2),
            "monthly_savings_usd": round(self.monthly_savings_usd, 2),
            "savings_percentage": round(self.savings_percentage, 1),
            "break_even_queries_per_day": self.break_even_queries_per_day,
            "break_even_days": round(self.break_even_days, 1),
            "recommended_strategy": self.recommended_strategy,
            "strategic_rationale": self.strategic_rationale,
            "latency_cloud_ms": round(self.latency_cloud_ms, 1),
            "latency_local_ms": round(self.latency_local_ms, 1),
        }


class DecisionAdvisor:
    """Evaluates deployment volumes and formulates actionable infrastructure decisions."""

    def __init__(self, hardware_params: LocalHardwareParams | None = None) -> None:
        self.hardware = hardware_params or LocalHardwareParams()

    def evaluate(self, projected_monthly_queries: int, avg_input_tokens: int = 1200,
                 avg_output_tokens: int = 350, cloud_model: str = "gpt-4o") -> DecisionAssessment:
        """Analyze workload economics and formulate a recommendation."""
        cost_per_query = calculate_cost(
            cloud_model, input_tokens=avg_input_tokens, output_tokens=avg_output_tokens
        )
        cloud_monthly = cost_per_query * projected_monthly_queries
        local_monthly = self.hardware.total_monthly_cost_usd

        savings = cloud_monthly - local_monthly
        pct_savings = (savings / cloud_monthly * 100.0) if cloud_monthly > 0 else 0.0

        # Break-even calculation
        # Cost per day local = local_monthly / 30
        daily_local = local_monthly / 30.0
        break_even_qpd = int(daily_local / cost_per_query) if cost_per_query > 0 else 10_000
        days_to_break_even = (self.hardware.hardware_cost_usd / max(0.01, (cloud_monthly - self.hardware.monthly_electricity_usd) / 30.0))

        # Latency estimates
        # Cloud: ~300ms network round-trip + 110ms compute = 410ms
        # Local: ~0ms network + 145ms PCIe GPU compute = 145ms
        cloud_latency = 410.0
        local_latency = 145.0

        if savings > 500.0 and projected_monthly_queries >= 50_000:
            strategy = "Dedicated Local Workstation"
            rationale = (
                f"Local on-device execution delivers an estimated {pct_savings:.1f}% monthly cost reduction "
                f"(${savings:,.2f} savings) and executes ~2.8x faster without network latency or data egress."
            )
        elif cloud_monthly < 50.0:
            strategy = "Cloud API Deployment"
            rationale = (
                "Workload volume is low enough that cloud API costs remain under the local workstation's "
                "fixed monthly amortization and electricity threshold."
            )
        else:
            strategy = "Hybrid Partitioning"
            rationale = (
                "Run high-frequency embeddings, preliminary extractions, and audio processing locally on-device "
                "to minimize bandwidth while delegating frontier multi-step reasoning to external cloud endpoints."
            )

        return DecisionAssessment(
            projected_monthly_queries=projected_monthly_queries,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            cloud_model=cloud_model,
            cloud_monthly_cost_usd=cloud_monthly,
            local_monthly_cost_usd=local_monthly,
            monthly_savings_usd=savings,
            savings_percentage=pct_savings,
            break_even_queries_per_day=break_even_qpd,
            break_even_days=max(1.0, days_to_break_even),
            recommended_strategy=strategy,
            strategic_rationale=rationale,
            latency_cloud_ms=cloud_latency,
            latency_local_ms=local_latency,
        )
