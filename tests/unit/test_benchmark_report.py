from __future__ import annotations

from noelle.app.ui.pages.comparative_page import markdown_report


def test_markdown_report_contains_matrix_and_outputs() -> None:
    report = {
        "run_id": "abc123",
        "created_at": "2026-11-23T14:00:00",
        "hardware": {"gpu": "Apple M2 Pro GPU", "usable_bytes": 10 * 1024**3},
        "prompt": "Explain KV caching.",
        "system_prompt": "Be brief.",
        "params": {"temperature": 0, "max_new_tokens": 64},
        "queue_plan": {"mode": "hot_swap"},
        "views": [
            {"view": 1, "model_id": "org/small", "status": "finished", "text": "Answer one.",
             "metrics": {"ttft_ms": 85.0, "tokens_per_second": 112.5, "model_peak_bytes": 3 * 1024**3,
                         "precision": "fp16", "engine": "transformers", "device": "mps", "load_ms": 900,
                         "completion_tokens": 40}},
            {"view": 2, "model_id": "org/big", "status": "error", "text": "",
             "error": {"message": "OOM"}, "metrics": {}},
        ],
    }
    text = markdown_report(report)
    assert "# Noelle benchmark abc123" in text
    assert "| Metric | View 1 (small) | View 2 (big) |" in text
    assert "| Time to first token | 85 ms | — |" in text
    assert "| Generation speed | 112.5 tok/s | — |" in text
    assert "| Peak memory | 3.0 GB | — |" in text
    assert "Queue mode: hot_swap" in text
    assert "Answer one." in text and "*error*" in text
    assert '"max_new_tokens": 64' in text
