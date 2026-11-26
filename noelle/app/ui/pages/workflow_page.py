"""Workflow Lab Page: Hybrid Cloud + Local Multi-Modal Composition & Benchmarking.

Allows users to compose, compare, and benchmark pipelines across local runtimes
and hosted cloud APIs with invariant telemetry, privacy guardrails, and decision modeling.
"""

from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noelle.app.context import AppContext
from noelle.app.ui.widgets.common import Banner, Panel, button, label
from noelle.evaluation.generators.tts_generator import generate_tts_suite
from noelle.providers.registry import list_providers
from noelle.services.cost_advisor import DecisionAdvisor
from noelle.workflow.graph import PrivacyMode, WorkflowGraph
from noelle.workflow.nodes import create_node
from noelle.workflow.runner import WorkflowResult, WorkflowRunner


class WorkflowPage(QWidget):
    """The central Hybrid Workflow & Evaluation Lab interface."""

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self.graph = WorkflowGraph(name="Voice Assistant Pipeline", privacy_mode=PrivacyMode.HYBRID)
        self.runner = WorkflowRunner(node_callback=self._on_node_step)
        self._advisor = DecisionAdvisor()
        self._selected_node_id: str | None = None

        self._build_ui()
        self._load_template("voice_assistant")

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        # Top Control & Privacy Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.addWidget(label("Workflow Lab", "h2"))
        title_box.addWidget(label("Hybrid Cloud + Local Multi-Modal Composition & Empirical Benchmarks", "muted"))
        header.addLayout(title_box, 1)

        # Privacy Mode Selector
        header.addWidget(label("Privacy Mode:", "strong"))
        self.privacy_combo = QComboBox()
        self.privacy_combo.addItem("🔒 Local-Only (Zero Egress)", PrivacyMode.LOCAL_ONLY)
        self.privacy_combo.addItem("⚡ Hybrid (Controlled Egress)", PrivacyMode.HYBRID)
        self.privacy_combo.addItem("☁ Cloud-Only (Full Hosted)", PrivacyMode.CLOUD_ONLY)
        self.privacy_combo.setCurrentIndex(1)
        self.privacy_combo.currentIndexChanged.connect(self._on_privacy_changed)
        header.addWidget(self.privacy_combo)

        # Template Selector
        header.addWidget(label("Template:", "faint"))
        self.template_combo = QComboBox()
        self.template_combo.addItem("Voice Assistant (STT → LLM → TTS)", "voice_assistant")
        self.template_combo.addItem("Hybrid Multimodal RAG", "multimodal_rag")
        self.template_combo.addItem("Local vs. Cloud LLM Compare", "llm_compare")
        self.template_combo.addItem("Embedding Retrieval Lab", "embedding_lab")
        self.template_combo.currentIndexChanged.connect(self._on_template_selected)
        header.addWidget(self.template_combo)

        # Execution Button
        self.run_btn = button("▶ Run Pipeline", "play", "primary")
        self.run_btn.clicked.connect(self._run_pipeline)
        header.addWidget(self.run_btn)

        root.addLayout(header)

        # Main Splitter (Left: Node Hierarchy & Palette | Right: Inspector & Tabs)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Panel: Node Palette and Graph Order
        left_panel = Panel()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.addWidget(label("Pipeline Nodes", "strong"))

        # Add Node Buttons
        palette_layout = QHBoxLayout()
        add_llm = button("+ LLM", "plus")
        add_llm.clicked.connect(lambda: self._add_node_type("llm"))
        add_embed = button("+ Embed", "plus")
        add_embed.clicked.connect(lambda: self._add_node_type("embedding"))
        add_stt = button("+ STT", "plus")
        add_stt.clicked.connect(lambda: self._add_node_type("stt"))
        add_tts = button("+ TTS", "plus")
        add_tts.clicked.connect(lambda: self._add_node_type("tts"))
        palette_layout.addWidget(add_llm)
        palette_layout.addWidget(add_embed)
        palette_layout.addWidget(add_stt)
        palette_layout.addWidget(add_tts)
        left_layout.addLayout(palette_layout)

        # Node List
        self.node_list = QListWidget()
        self.node_list.currentItemChanged.connect(self._on_node_selected)
        left_layout.addWidget(self.node_list, 1)

        # Node control buttons
        node_btns = QHBoxLayout()
        remove_btn = button("Remove Node", "trash", "danger")
        remove_btn.clicked.connect(self._remove_selected_node)
        node_btns.addWidget(remove_btn)
        left_layout.addLayout(node_btns)

        splitter.addWidget(left_panel)

        # Right Panel: Tabbed Workspaces (Inspector, Telemetry, Decision Advisor, Test Generator)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_inspector_tab(), "Node Inspector")
        tabs.addTab(self._build_telemetry_tab(), "Results & Telemetry")
        tabs.addTab(self._build_decision_advisor_tab(), "Decision Advisor (TCO)")
        tabs.addTab(self._build_adversarial_tab(), "Adversarial Test Suite")
        self.tabs = tabs
        splitter.addWidget(tabs)
        splitter.setSizes([320, 680])

        root.addWidget(splitter, 1)

    # ---------------------------------------------------------------- Node Inspector
    def _build_inspector_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        self.inspector_banner = Banner("info", "Select a node on the left to configure parameters and execution target.")
        layout.addWidget(self.inspector_banner)

        form = QFormLayout()
        self.node_title_edit = QLineEdit()
        self.node_title_edit.textChanged.connect(self._save_inspector_changes)
        form.addRow("Title:", self.node_title_edit)

        self.provider_combo = QComboBox()
        for p in list_providers():
            self.provider_combo.addItem(f"{p.name} ({'Local' if p.is_local else 'Cloud'})", p.id)
        self.provider_combo.currentIndexChanged.connect(self._save_inspector_changes)
        form.addRow("Target Provider:", self.provider_combo)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("Model identifier (e.g. gpt-4o, BAAI/bge-m3)")
        self.model_edit.textChanged.connect(self._save_inspector_changes)
        form.addRow("Model ID:", self.model_edit)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Input prompt, template, or text payload...")
        self.prompt_edit.textChanged.connect(self._save_inspector_changes)
        form.addRow("Prompt / Content:", self.prompt_edit)

        layout.addLayout(form)
        layout.addStretch(1)
        return page

    # ---------------------------------------------------------------- Results & Telemetry
    def _build_telemetry_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        # Summary Cards
        cards_layout = QHBoxLayout()
        self.stat_latency = label("Latency: —", "strong")
        self.stat_cost = label("Cost: —", "strong")
        self.stat_tokens = label("Tokens: —", "strong")
        self.stat_speed = label("Speed: —", "strong")
        cards_layout.addWidget(self.stat_latency)
        cards_layout.addWidget(self.stat_cost)
        cards_layout.addWidget(self.stat_tokens)
        cards_layout.addWidget(self.stat_speed)
        layout.addLayout(cards_layout)

        # Step breakdown table
        self.telemetry_table = QTableWidget(0, 5)
        self.telemetry_table.setHorizontalHeaderLabels(["Node", "Provider", "Latency", "Cost", "Tokens"])
        self.telemetry_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.telemetry_table, 1)

        # Output payload viewer
        layout.addWidget(label("Output Payload:", "strong"))
        self.output_view = QPlainTextEdit()
        self.output_view.setReadOnly(True)
        layout.addWidget(self.output_view, 1)

        return page

    # ---------------------------------------------------------------- Decision Advisor Tab
    def _build_decision_advisor_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(label("Cloud vs. Local Infrastructure Decision Advisor", "h3"))
        layout.addWidget(label("Predictive TCO modeling comparing hosted cloud tokens against dedicated workstation amortization and power.", "muted"))

        form = QFormLayout()
        self.queries_spin = QSpinBox()
        self.queries_spin.setRange(1_000, 50_000_000)
        self.queries_spin.setSingleStep(50_000)
        self.queries_spin.setValue(500_000)
        self.queries_spin.valueChanged.connect(self._recalculate_decision)
        form.addRow("Monthly Queries:", self.queries_spin)

        self.input_tokens_spin = QSpinBox()
        self.input_tokens_spin.setRange(10, 32_000)
        self.input_tokens_spin.setValue(1200)
        self.input_tokens_spin.valueChanged.connect(self._recalculate_decision)
        form.addRow("Avg Input Tokens / Query:", self.input_tokens_spin)

        self.output_tokens_spin = QSpinBox()
        self.output_tokens_spin.setRange(10, 8_000)
        self.output_tokens_spin.setValue(350)
        self.output_tokens_spin.valueChanged.connect(self._recalculate_decision)
        form.addRow("Avg Output Tokens / Query:", self.output_tokens_spin)

        layout.addLayout(form)

        # Decision Summary Banner
        self.decision_banner = Banner("info", "Calculating...")
        layout.addWidget(self.decision_banner)

        # Comparison Table
        self.decision_table = QTableWidget(7, 3)
        self.decision_table.setHorizontalHeaderLabels(["Metric Dimension", "Cloud API (OpenAI)", "Local Workstation (RTX 4090)"])
        self.decision_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.decision_table, 1)

        self._recalculate_decision()
        return page

    # ---------------------------------------------------------------- Adversarial Test Tab
    def _build_adversarial_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(label("Adversarial Test Suite Generator", "h3"))
        layout.addWidget(label("Generates realistic, domain-specific evaluation cases targeting difficult pronunciation, currency numbers, and technical terminology in Indonesian.", "muted"))

        btn_row = QHBoxLayout()
        gen_tts_btn = button("Generate Indonesian TTS Suite", "sparkles", "primary")
        gen_tts_btn.clicked.connect(self._generate_tts_suite)
        btn_row.addWidget(gen_tts_btn)
        layout.addLayout(btn_row)

        self.test_suite_list = QListWidget()
        layout.addWidget(self.test_suite_list, 1)

        return page

    # ---------------------------------------------------------------- Logic & Handlers
    def _on_privacy_changed(self) -> None:
        mode = self.privacy_combo.currentData()
        self.graph.privacy_mode = mode
        self._refresh_node_list()

    def _on_template_selected(self) -> None:
        tid = self.template_combo.currentData()
        self._load_template(tid)

    def _load_template(self, template_id: str) -> None:
        self.graph = WorkflowGraph(name=template_id, privacy_mode=self.privacy_combo.currentData())

        if template_id == "voice_assistant":
            n1 = create_node("input", "in_1", title="Microphone Input", parameters={"value": "Transkripsikan laporan fiskal 2026."})
            n2 = create_node("stt", "stt_1", title="Whisper Local", provider_id="local", model_id="openai/whisper-large-v3")
            n3 = create_node("llm", "llm_1", title="Qwen 2.5 Cloud", provider_id="openrouter", model_id="qwen/qwen-2.5-72b-instruct")
            n4 = create_node("tts", "tts_1", title="Local SpeechT5", provider_id="local", model_id="microsoft/speecht5_tts")
            self.graph.add_node(n1)
            self.graph.add_node(n2)
            self.graph.add_node(n3)
            self.graph.add_node(n4)
            self.graph.connect("in_1", "output", "stt_1", "audio")
            self.graph.connect("stt_1", "transcript", "llm_1", "prompt")
            self.graph.connect("llm_1", "text", "tts_1", "text")

        elif template_id == "embedding_lab":
            n1 = create_node("input", "in_1", title="Corpus Ingestion", parameters={"value": "Analisis dampak AI lokal terhadap efisiensi infrastruktur."})
            n2 = create_node("embedding", "emb_local", title="Local BGE-M3", provider_id="local", model_id="BAAI/bge-m3")
            n3 = create_node("embedding", "emb_cloud", title="OpenAI Embeddings", provider_id="openai", model_id="text-embedding-3-small")
            self.graph.add_node(n1)
            self.graph.add_node(n2)
            self.graph.add_node(n3)
            self.graph.connect("in_1", "output", "emb_local", "text")
            self.graph.connect("in_1", "output", "emb_cloud", "text")

        elif template_id == "multimodal_rag":
            n1 = create_node("input", "in_1", title="Invoice Document", parameters={"value": "Berapa total tagihan invoice ini?"})
            n2 = create_node("vision", "vis_1", title="Cloud Vision (GPT-4o)", provider_id="openai", model_id="gpt-4o")
            n3 = create_node("rag", "rag_1", title="Local Knowledge Index", provider_id="local", parameters={"top_k": 3})
            n4 = create_node("llm", "llm_1", title="Local Reasoning (Qwen)", provider_id="local", model_id="Qwen/Qwen2.5-7B-Instruct")
            self.graph.add_node(n1)
            self.graph.add_node(n2)
            self.graph.add_node(n3)
            self.graph.add_node(n4)
            self.graph.connect("in_1", "output", "vis_1", "prompt")
            self.graph.connect("vis_1", "text", "rag_1", "query")
            self.graph.connect("rag_1", "context", "llm_1", "context")

        self._refresh_node_list()

    def _refresh_node_list(self) -> None:
        self.node_list.clear()
        for nid, node in self.graph.nodes.items():
            badge = "🟢 Local" if not node.is_cloud else f"☁ {node.provider_id.upper()}"
            item = QListWidgetItem(f"{node.title}  [{badge}]")
            item.setData(Qt.ItemDataRole.UserRole, nid)
            self.node_list.addItem(item)

        if self.node_list.count() > 0:
            self.node_list.setCurrentRow(0)

    def _add_node_type(self, ntype: str) -> None:
        idx = len(self.graph.nodes) + 1
        nid = f"{ntype}_{idx}"
        node = create_node(ntype, nid, title=f"New {ntype.upper()} {idx}")
        self.graph.add_node(node)
        self._refresh_node_list()

    def _remove_selected_node(self) -> None:
        if self._selected_node_id:
            self.graph.remove_node(self._selected_node_id)
            self._selected_node_id = None
            self._refresh_node_list()

    def _on_node_selected(self, item: QListWidgetItem | None) -> None:
        if not item:
            return
        nid = item.data(Qt.ItemDataRole.UserRole)
        self._selected_node_id = nid
        node = self.graph.nodes.get(nid)
        if not node:
            return

        self.node_title_edit.setText(node.title)
        idx = self.provider_combo.findData(node.provider_id)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)
        self.model_edit.setText(node.model_id)
        self.prompt_edit.setPlainText(str(node.parameters.get("value") or node.parameters.get("prompt") or ""))

    def _save_inspector_changes(self) -> None:
        if not self._selected_node_id:
            return
        node = self.graph.nodes.get(self._selected_node_id)
        if not node:
            return

        node.title = self.node_title_edit.text()
        node.provider_id = self.provider_combo.currentData() or "local"
        node.model_id = self.model_edit.text()
        text_val = self.prompt_edit.toPlainText()
        if node.node_type == "input":
            node.parameters["value"] = text_val
        else:
            node.parameters["prompt"] = text_val

    def _run_pipeline(self) -> None:
        # Pre-dispatch Privacy Check
        if self.graph.privacy_mode == PrivacyMode.HYBRID and self.graph.has_cloud_nodes():
            cloud_nodes = self.graph.get_cloud_nodes()
            names = ", ".join(f"{n.title} ({n.provider_id})" for n in cloud_nodes)
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Confirm Cloud Egress")
            dlg.setIcon(QMessageBox.Icon.Warning)
            dlg.setText(f"This workflow will dispatch data to external cloud APIs:\n\n{names}\n\nProceed?")
            dlg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if dlg.exec() != QMessageBox.StandardButton.Ok:
                return

        from noelle.app.services.task_runner import runner

        graph = self.graph
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")

        def work() -> WorkflowResult:
            # A fresh runner per run, without Qt callbacks: it executes on a worker thread.
            return WorkflowRunner().execute(graph, consent_given=True)

        def done(res: WorkflowResult) -> None:
            self._finish_run()
            if not res.is_success:
                QMessageBox.critical(self, "Pipeline Error", res.error_message or "Execution failed.")
                return
            self.stat_latency.setText(f"Latency: {res.total_latency_ms:.1f} ms")
            self.stat_cost.setText(f"Cost: ${res.total_cost_usd:.6f}")
            self.stat_tokens.setText(f"Tokens: {res.total_tokens}")
            self.telemetry_table.setRowCount(0)
            for nid, tel in res.telemetry.items():
                node = self.graph.nodes.get(nid)
                row = self.telemetry_table.rowCount()
                self.telemetry_table.insertRow(row)
                self.telemetry_table.setItem(row, 0, QTableWidgetItem(node.title if node else nid))
                self.telemetry_table.setItem(row, 1, QTableWidgetItem(node.provider_id if node else "—"))
                self.telemetry_table.setItem(row, 2, QTableWidgetItem(f"{tel.latency_ms:.1f} ms"))
                self.telemetry_table.setItem(row, 3, QTableWidgetItem(f"${tel.cost_usd:.6f}"))
                self.telemetry_table.setItem(row, 4, QTableWidgetItem(str(tel.input_tokens + tel.output_tokens)))
            self.output_view.setPlainText(json.dumps(res.to_dict()["node_outputs"], indent=2))
            self.tabs.setCurrentIndex(1)

        def failed(message: str) -> None:
            self._finish_run()
            QMessageBox.critical(self, "Pipeline Error", message)

        runner().submit(work, on_result=done, on_error=failed)

    def _finish_run(self) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶ Run Pipeline")

    def _on_node_step(self, node_id: str, status: str) -> None:
        pass

    def _recalculate_decision(self) -> None:
        queries = self.queries_spin.value()
        in_toks = self.input_tokens_spin.value()
        out_toks = self.output_tokens_spin.value()
        assessment = self._advisor.evaluate(queries, in_toks, out_toks)

        self.decision_banner.set_text(
            f"Recommended Strategy: {assessment.recommended_strategy}\n{assessment.strategic_rationale}"
        )

        rows = [
            ("Direct API Costs", f"${assessment.cloud_monthly_cost_usd:,.2f} / month", "$0.00 / month"),
            ("Local Electricity Overhead", "$0.00 / month", f"${self._advisor.hardware.monthly_electricity_usd:,.2f} / month"),
            ("Hardware Amortization", "$0.00 / month", f"${self._advisor.hardware.monthly_amortization_usd:,.2f} / month"),
            ("Total Monthly Net Cost", f"${assessment.cloud_monthly_cost_usd:,.2f} / month", f"${assessment.local_monthly_cost_usd:,.2f} / month"),
            ("Monthly Net Savings", f"— (${assessment.monthly_savings_usd:,.2f} difference)", f"${assessment.monthly_savings_usd:,.2f} ({assessment.savings_percentage:.1f}%)"),
            ("Average Latency", f"{assessment.latency_cloud_ms:.1f} ms (network + compute)", f"{assessment.latency_local_ms:.1f} ms (PCIe direct)"),
            ("Break-Even Threshold", f"{assessment.break_even_queries_per_day:,} queries / day", f"Parity reached in {assessment.break_even_days:.1f} days"),
        ]

        for r_idx, (dimension, cloud_val, local_val) in enumerate(rows):
            self.decision_table.setItem(r_idx, 0, QTableWidgetItem(dimension))
            self.decision_table.setItem(r_idx, 1, QTableWidgetItem(cloud_val))
            self.decision_table.setItem(r_idx, 2, QTableWidgetItem(local_val))

    def _generate_tts_suite(self) -> None:
        suite = generate_tts_suite(count=10)
        self.test_suite_list.clear()
        for tc in suite:
            challenges = ", ".join(tc.expected_challenges)
            item = QListWidgetItem(f"[{tc.category.upper()}]  {tc.text}\n  ↳ Challenges: {challenges}")
            self.test_suite_list.addItem(item)
