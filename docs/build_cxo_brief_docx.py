"""Build CXO briefing DOCX for FinnAI SLM fine-tuning (with flowcharts)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

OUT_DIR = Path(__file__).resolve().parent
IMG = OUT_DIR / "_figures"
IMG.mkdir(exist_ok=True)


def _set_run_font(run, size=11, bold=False, color=None):
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x0F, 0x3D, 0x3E)
    return h


def add_para(doc, text, size=11, bold=False, space_after=8):
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, size=size, bold=bold)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    return p


def add_callout(doc, title, body, fill="E8F4F3"):
    """Simple shaded paragraph block via table cell."""
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shading)
    p0 = cell.paragraphs[0]
    r0 = p0.add_run(title)
    _set_run_font(r0, size=11, bold=True, color=(0x0F, 0x3D, 0x3E))
    p1 = cell.add_paragraph()
    r1 = p1.add_run(body)
    _set_run_font(r1, size=10)
    doc.add_paragraph()


def add_image(doc, path: Path, width=6.2, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        _set_run_font(r, size=9, color=(0x55, 0x55, 0x55))
        r.italic = True


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(h)
        _set_run_font(run, size=10, bold=True, color=(0xFF, 0xFF, 0xFF))
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "0F3D3E")
        cell._tc.get_or_add_tcPr().append(shading)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            _set_run_font(run, size=10)
    doc.add_paragraph()


# ---------- figures ----------

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    boxes = [
        (0.3, 1.5, "1. Data\nSynthetic SMS\n+ coach Q&A\n+ Indic expand", "#D6EFEA"),
        (2.6, 1.5, "2. Split\nTrain / Val / Test\nby template\n(no leakage)", "#C8E6C9"),
        (4.9, 1.5, "3. QLoRA SFT\nQwen3-1.7B\n3 epochs · A10G\n~5 hours", "#BBDEFB"),
        (7.2, 1.5, "4. Evaluate\nPre-registered\nship gates\nR-EM · chat", "#FFE0B2"),
        (9.3, 1.5, "5. Ship\nINT4 on phone\n+ open-source\nHF + GitHub", "#F8BBD0"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), 1.9, 2.0, boxstyle="round,pad=0.04,rounding_size=0.15",
                facecolor=color, edgecolor="#333333", linewidth=1.2,
            )
        )
        ax.text(x + 0.95, y + 1.0, text, ha="center", va="center", fontsize=8.5, fontweight="medium")
    for i in range(4):
        x0 = boxes[i][0] + 1.9
        x1 = boxes[i + 1][0]
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5),
                    arrowprops=dict(arrowstyle="->", color="#0F3D3E", lw=1.8))
    ax.set_title("FinnAI SLM — end-to-end fine-tuning pipeline", fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    path = IMG / "pipeline.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def fig_decision():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, color, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=color, edgecolor="#333", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    box(3.2, 4.5, 3.6, 0.7, "Is user financial data sensitive?", "#FFF9C4", 10)
    ax.annotate("", xy=(1.8, 3.6), xytext=(4.2, 4.5), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(8.2, 3.6), xytext=(5.8, 4.5), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(2.5, 4.05, "Yes", fontsize=8, color="#C62828")
    ax.text(7.0, 4.05, "No / low sensitivity", fontsize=8, color="#2E7D32")

    box(0.4, 2.9, 2.8, 0.8, "On-device required\n(privacy + offline)", "#FFCDD2")
    box(7.0, 2.9, 2.6, 0.8, "Cloud LLM API\nis often enough", "#C8E6C9")

    ax.annotate("", xy=(1.8, 2.2), xytext=(1.8, 2.9), arrowprops=dict(arrowstyle="->", color="#333"))
    box(0.3, 1.2, 3.0, 1.0, "Can you build 5k–16k\nlabeled / synthetic examples?", "#E1BEE7")
    ax.annotate("", xy=(1.8, 0.55), xytext=(1.8, 1.2), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(2.1, 0.9, "Yes", fontsize=8)
    box(0.3, 0.15, 3.0, 0.55, "→ Fine-tune small SLM (FinnAI path)", "#BBDEFB", 9)

    ax.annotate("", xy=(5.5, 1.7), xytext=(3.3, 1.7), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.text(3.8, 1.9, "No", fontsize=8)
    box(5.5, 1.2, 4.0, 1.0, "Few-shot with cloud LLM,\nthen bootstrap synthetic data\n→ later fine-tune", "#FFE0B2")

    ax.set_title("When does fine-tuning a small on-device model make sense?", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = IMG / "decision.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def fig_qlora():
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.add_patch(FancyBboxPatch((0.4, 0.8), 4.0, 3.0, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#E3F2FD", edgecolor="#1565C0", lw=1.5))
    ax.text(2.4, 3.5, "Base model (frozen)", ha="center", fontsize=11, fontweight="bold", color="#1565C0")
    ax.text(2.4, 2.4, "Qwen3-1.7B\n~1.7 billion parameters\n\nKept mostly unchanged\n(stored in 4-bit during train)",
            ha="center", va="center", fontsize=9)

    ax.add_patch(FancyBboxPatch((5.6, 0.8), 4.0, 3.0, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#FFF3E0", edgecolor="#EF6C00", lw=1.5))
    ax.text(7.6, 3.5, "LoRA adapter (trained)", ha="center", fontsize=11, fontweight="bold", color="#EF6C00")
    ax.text(7.6, 2.4, "Only ~67 MB of new weights\n(rank 16 matrices)\n\nLearns SMS JSON style +\nfinance-coach groundedness",
            ha="center", va="center", fontsize=9)

    ax.annotate("", xy=(5.6, 2.3), xytext=(4.4, 2.3),
                arrowprops=dict(arrowstyle="<->", color="#0F3D3E", lw=2))
    ax.text(5.0, 2.6, "QLoRA", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("QLoRA in one picture — teach a specialist without rebuilding the brain", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = IMG / "qlora.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 4.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor="#FAFAFA", edgecolor="#0F3D3E", lw=1.5))
    ax.text(5.0, 4.6, "FinnDot Android app (on the phone)", ha="center", fontsize=12, fontweight="bold")

    ax.add_patch(FancyBboxPatch((0.6, 3.2), 2.6, 1.0, boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor="#E8F5E9", edgecolor="#2E7D32"))
    ax.text(1.9, 3.7, "SMS inbox", ha="center", fontsize=10, fontweight="bold")

    ax.add_patch(FancyBboxPatch((3.6, 3.2), 3.0, 1.0, boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor="#E3F2FD", edgecolor="#1565C0"))
    ax.text(5.1, 3.7, "parser-core\n35+ bank regex parsers", ha="center", fontsize=9)

    ax.add_patch(FancyBboxPatch((7.0, 3.2), 2.4, 1.0, boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor="#FCE4EC", edgecolor="#AD1457"))
    ax.text(8.2, 3.7, "Known bank?\nYes → done", ha="center", fontsize=9)

    ax.annotate("", xy=(3.6, 3.7), xytext=(3.2, 3.7), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(7.0, 3.7), xytext=(6.6, 3.7), arrowprops=dict(arrowstyle="->", color="#333"))

    ax.add_patch(FancyBboxPatch((2.5, 1.4), 5.0, 1.4, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor="#FFF8E1", edgecolor="#F9A825", lw=1.5))
    ax.text(5.0, 2.35, "FinnAI SLM (on-device)", ha="center", fontsize=11, fontweight="bold")
    ax.text(5.0, 1.8, "Unknown banks · Indic SMS · finance coach\nINT4 LiteRT-LM · ~0.9 GB · no cloud",
            ha="center", fontsize=9)

    ax.annotate("", xy=(5.0, 2.8), xytext=(5.1, 3.2),
                arrowprops=dict(arrowstyle="->", color="#AD1457", lw=1.5))
    ax.text(5.8, 3.0, "No → fallback", fontsize=8, color="#AD1457")

    ax.add_patch(FancyBboxPatch((0.6, 0.5), 4.0, 0.7, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#E0F7FA", edgecolor="#00838F"))
    ax.text(2.6, 0.85, "User Ask / Learn chat", ha="center", fontsize=9)
    ax.annotate("", xy=(3.5, 1.4), xytext=(2.6, 1.2), arrowprops=dict(arrowstyle="->", color="#00838F"))

    ax.set_title("Product architecture — regex first, SLM for the long tail + coaching", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = IMG / "architecture.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def fig_data_mix():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    sizes = [60, 25, 15]
    labels = ["SMS JSON\n60%", "Finance coach\n25%", "General finance\n15%"]
    colors = ["#26A69A", "#42A5F5", "#FFA726"]
    axes[0].pie(sizes, labels=labels, colors=colors, startangle=90,
                wedgeprops=dict(width=0.55, edgecolor="white"))
    axes[0].set_title("Training mix (16,000 examples)", fontsize=11, fontweight="bold")

    cats = ["Parser gold", "Synthetic\nEN SMS", "Indic SMS\n(Nova)", "Coach\nQ&A", "General\nQ&A", "Negatives"]
    vals = [3000, 6000, 300, 4000, 2400, 300]
    bars = axes[1].barh(cats, vals, color="#0F3D3E")
    axes[1].set_xlabel("Approx. examples")
    axes[1].set_title("Where the data came from (synthetic)", fontsize=11, fontweight="bold")
    axes[1].invert_yaxis()
    for b, v in zip(bars, vals):
        axes[1].text(v + 80, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=8)

    fig.tight_layout()
    path = IMG / "data_mix.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def fig_results():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    fig.patch.set_facecolor("white")
    models = ["Qwen3-1.7B\n(base)", "Qwen2.5-1.5B\n(old product)", "FinnAI v1", "FinnAI v2"]
    rem = [28.16, 42.43, 92.75, 97.97]
    colors = ["#90A4AE", "#78909C", "#26A69A", "#0F3D3E"]
    bars = ax.bar(models, rem, color=colors, width=0.65)
    ax.set_ylabel("SMS Strict Record Exact-Match %")
    ax.set_ylim(0, 110)
    ax.set_title("Held-out SMS accuracy — fine-tuning closed the gap", fontsize=12, fontweight="bold")
    for b, v in zip(bars, rem):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(95, color="#C62828", ls="--", lw=1, alpha=0.7)
    ax.text(3.4, 96.5, "practical target", fontsize=8, color="#C62828")
    fig.tight_layout()
    path = IMG / "results.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def fig_privacy():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    items = [
        (0.4, "User SMS\nNEVER used", "#FFCDD2"),
        (2.7, "Synthetic\ntemplates only", "#C8E6C9"),
        (5.0, "Gold labels\nowned by us", "#BBDEFB"),
        (7.3, "Open-source\nApache 2.0", "#FFE0B2"),
    ]
    for x, text, color in items:
        ax.add_patch(FancyBboxPatch((x, 1.0), 2.1, 2.0, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    facecolor=color, edgecolor="#333", lw=1.1))
        ax.text(x + 1.05, 2.0, text, ha="center", va="center", fontsize=10, fontweight="medium")
    ax.set_title("Privacy & ownership — why this can be open-sourced safely", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = IMG / "privacy.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def build_doc(paths: dict[str, Path]) -> Path:
    doc = Document()

    # Margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # Cover
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("FinnAI SLM")
    _set_run_font(r, size=28, bold=True, color=(0x0F, 0x3D, 0x3E))

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = st.add_run("How we fine-tuned an on-device finance AI — for CXOs and builders")
    _set_run_font(r, size=14, color=(0x33, 0x33, 0x33))

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = meta.add_run(
        "FinnDot · September 2026 · Apache 2.0\n"
        "Model: https://huggingface.co/finndot/finnai-slm-v2\n"
        "Dataset: https://huggingface.co/datasets/finndot/finnai-slm-data\n"
        "Code: https://github.com/arunbhatg/finnai-slm"
    )
    _set_run_font(r, size=10, color=(0x55, 0x55, 0x55))

    add_callout(
        doc,
        "How to read this brief",
        "Plain English sections are for CXOs and product leaders who use ChatGPT-like tools but do not train models. "
        "Technical callouts keep the real method, metrics, and trade-offs — nothing important is hidden. "
        "You can skim the teal boxes and charts first, then dive into the technical parts where needed.",
    )

    # 1. Executive summary
    add_heading(doc, "1. Executive summary", 1)
    add_para(
        doc,
        "FinnDot automatically turns Indian bank SMS into expense records and answers money questions. "
        "Regex parsers already cover 35+ banks. The remaining hard cases — unknown banks, Hindi/Hinglish SMS, "
        "and finance coaching — need language understanding.",
    )
    add_para(
        doc,
        "We fine-tuned a small open-source model (Qwen3, 1.7 billion parameters) so it runs on the user's phone. "
        "After training, SMS field accuracy rose from ~28% (base model) to ~98% (FinnAI v2) on a held-out test set. "
        "Training data is synthetic — no real user SMS. Total project GPU + data cost was on the order of tens of dollars, "
        "not millions.",
    )
    add_callout(
        doc,
        "One-line takeaway",
        "A narrow, privacy-critical product job does not need a giant cloud model. "
        "A carefully fine-tuned small model can beat a general phone-sized baseline and ship under Apache 2.0.",
    )
    add_image(doc, paths["results"], caption="Figure 1 — SMS extraction accuracy before vs after fine-tuning")

    # 2. Audience map
    add_heading(doc, "2. Two audiences, one story", 1)
    add_table(
        doc,
        ["If you are…", "Focus on…", "Skip / skim…"],
        [
            ["CXO / product (LLM as a user)", "§1, §3, §4 charts, §8 cost, §9 open-source", "Hyperparameter tables"],
            ["Tech lead / ML-curious", "§5–§7 technical callouts + eval gates", "Nothing — read all"],
            ["Partner / investor", "§1, §4 privacy, §8 cost, HF/GitHub links", "AWS instance types"],
        ],
    )

    # 3. Business problem
    add_heading(doc, "3. The business problem (plain English)", 1)
    add_para(
        doc,
        "Banks text customers after every UPI payment. FinnDot reads those texts and books the expense automatically. "
        "That saves users from typing. It only works if we can reliably extract: amount, merchant, type, account last-4, balance.",
    )
    add_para(doc, "Example SMS", bold=True)
    add_para(doc, "HDFC Bank: Rs.499.00 debited from A/c XX1234 to SWIGGY via UPI. Avl Bal Rs.15000.50", size=10)
    add_para(doc, "Target machine-readable result", bold=True)
    add_para(
        doc,
        '{"amount": 499.0, "merchant": "SWIGGY", "type": "EXPENSE", "account": "1234", "balance": 15000.5, "category": "Food"}',
        size=10,
    )

    add_heading(doc, "3.1 Why not only rules (regex)?", 2)
    add_para(
        doc,
        "Rules are excellent for known banks — we keep them. They fail on the long tail: new banks, format changes, "
        "Indic scripts, and OTP/promo messages that must be ignored. Coaching (“Where am I overspending?”) is not a regex problem.",
    )

    add_heading(doc, "3.2 Why not only ChatGPT-in-the-cloud?", 2)
    add_para(
        doc,
        "Bank SMS contains balances and account fragments. Sending every message to a cloud API creates privacy, cost, "
        "latency, and offline gaps. FinnDot’s product promise is local-first AI when the user wants it.",
    )
    add_image(doc, paths["decision"], caption="Figure 2 — Decision path: when on-device fine-tuning is the right bet")

    # 4. What is fine-tuning
    add_heading(doc, "4. What “fine-tuning” means (for non-ML readers)", 1)
    add_para(
        doc,
        "Think of a large language model as a very well-read generalist. Fine-tuning is like sending that generalist "
        "on a short, intense apprenticeship for one job — reading Indian bank SMS and answering finance questions with numbers "
        "that appear in the user’s ledger — without rewriting everything they already know.",
    )
    add_callout(
        doc,
        "Analogy",
        "Hiring a brilliant generalist and giving them a 5-hour bootcamp with 16,000 graded examples, "
        "instead of hiring a 200-person team or calling a consultant for every SMS.",
    )
    add_image(doc, paths["qlora"], caption="Figure 3 — QLoRA: freeze the base brain, train a small specialist adapter")

    add_heading(doc, "4.1 Technical: QLoRA SFT (do not skip)", 2)
    add_para(
        doc,
        "We used Supervised Fine-Tuning (SFT) with QLoRA on Qwen/Qwen3-1.7B (Apache 2.0). "
        "The base weights are quantized to 4-bit (NF4) during training so one NVIDIA A10G (24 GB) is enough. "
        "LoRA inserts low-rank matrices (rank 16, alpha 32) into attention and MLP projections "
        "(q, k, v, o, gate, up, down). Only those matrices train.",
    )
    add_table(
        doc,
        ["Knob", "Value", "Why it matters"],
        [
            ["Base model", "Qwen3-1.7B Instruct", "Open license + LiteRT on-device path"],
            ["Method", "QLoRA SFT", "Cheap, reversible, low forgetting"],
            ["Epochs", "3", "Enough for 16k specialized rows"],
            ["LR", "2e-4 cosine", "Standard QLoRA SFT setting"],
            ["Effective batch", "16", "Stable JSON formatting"],
            ["Max sequence", "1536 tokens", "Fits coach system + SMS"],
            ["Thinking", "Disabled", "Phone latency; SMS needs direct JSON"],
            ["Checkpoint pick", "Best val SMS R-EM", "Not lowest train loss"],
            ["Train cost", "~$8 / run", "Hours on g5.2xlarge, not weeks"],
        ],
    )

    # 5. Architecture
    add_heading(doc, "5. Product architecture", 1)
    add_para(
        doc,
        "FinnAI SLM is a fallback and a coach — not a replacement for deterministic parsers on known banks.",
    )
    add_image(doc, paths["architecture"], caption="Figure 4 — Regex first; SLM for unknown / Indic / coaching")

    # 6. Pipeline
    add_heading(doc, "6. How we did it — the pipeline", 1)
    add_image(doc, paths["pipeline"], width=6.4, caption="Figure 5 — Five stages from data to ship")

    add_heading(doc, "6.1 Stage 1–2: Privacy-safe data", 2)
    add_para(
        doc,
        "Plain English: We never trained on real customer SMS. We wrote templates that look like bank messages, "
        "filled them with fake amounts and merchants, and generated the correct JSON ourselves — so labels are free and clean.",
    )
    add_image(doc, paths["data_mix"], caption="Figure 6 — Training mix and data sources")
    add_image(doc, paths["privacy"], caption="Figure 7 — Privacy and ownership pillars")

    add_callout(
        doc,
        "Technical: split hygiene",
        "The split unit is (bank, template_id), not a single SMS line. All amount/merchant variants of one template "
        "stay in train OR val OR test. Seed 42. Target mix train-only: 60% SMS / 25% coach / 15% general. "
        "Chat eval is a frozen 50-item set never used in SFT. Nova Pro was used only as a surface-text factory "
        "for Indic paraphrases; rows were kept only if amount/merchant/last-4/ledger digits still matched our gold.",
    )

    add_heading(doc, "6.2 Stage 3: Training run", 2)
    add_para(
        doc,
        "Hardware: AWS us-east-2 (Ohio) g5.2xlarge (1× A10G) because Mumbai GPU quota was zero at the time. "
        "v2 trained on 16,000 examples. Validation SMS R-EM stayed at 0.965 across all three epochs (stable).",
    )

    add_heading(doc, "6.3 Stage 4: Evaluation — ship gates frozen first", 2)
    add_para(
        doc,
        "Plain English: Before looking at final scores, we wrote the pass/fail rules. That stops “it looks good enough” bias.",
    )
    add_table(
        doc,
        ["Gate", "Rule", "v2 result"],
        [
            ["1. SMS R-EM lift", "≥ +3 pp vs Qwen3-base, CI excludes 0", "Pass (+69.8 pp)"],
            ["2. Amount EM", "Drop vs base ≤ 1 pp", "Pass (+16.7 pp)"],
            ["3. Chat groundedness", "Drop vs base ≤ 5 pp", "Pass (+42 pp)"],
            ["4. JSON valid", "≥ 95%", "Pass (100%)"],
            ["5. On-device cost", "TTFT/RSS ≤ 1.3× old model", "Pending device bench"],
            ["RQ3 product", "R-EM ≥ Qwen2.5-1.5B", "Pass (97.97 > 42.43)"],
        ],
    )
    add_callout(
        doc,
        "Technical: primary metric",
        "Strict Record Exact-Match (R-EM) requires amount + type + merchant + account last-4 + balance all correct "
        "(or {} for non-transactions). We also report amount EM, merchant exact, field micro-F1, false-parse rate, "
        "bootstrap 95% CIs (10k resamples), and McNemar p vs base.",
    )

    add_heading(doc, "6.4 Stage 5: On-device + open source", 2)
    add_para(
        doc,
        "Production path: merge LoRA → export INT4 LiteRT-LM (~0.9 GB) → CloudFront for the app. "
        "Scientific copy: Hugging Face model + dataset + GitHub toolkit so others can reproduce or adapt.",
    )

    # 7. Results
    add_heading(doc, "7. Results that matter to the business", 1)
    add_table(
        doc,
        ["Metric", "FinnAI v2", "v1", "Qwen3 base"],
        [
            ["SMS R-EM %", "97.97", "92.75", "28.16"],
            ["Amount EM %", "99.53", "94.60", "82.84"],
            ["Merchant exact %", "98.21", "93.12", "61.31"],
            ["False-parse % (lower better)", "0.54", "28.57", "100"],
            ["Chat groundedness %", "68.0", "32.0", "26.0"],
        ],
    )
    add_para(
        doc,
        "Business reading: v2 almost eliminated “OTP treated as a payment” style mistakes (false-parse 28% → 0.5%) "
        "and more than doubled grounded coach answers. That is user trust, not just a leaderboard number.",
    )

    # 8. Cost
    add_heading(doc, "8. Cost & time (CXO view)", 1)
    add_table(
        doc,
        ["Item", "Approx. cost"],
        [
            ["One train run (g5.2xlarge ~5h)", "~$7–8"],
            ["Held-out eval (~3 models)", "~$6"],
            ["Indic data expansion (Nova API)", "~$2–3"],
            ["Full v1+v2 project incl. debug", "~$40–50"],
            ["Annotation army / user SMS", "$0 (not used)"],
        ],
    )
    add_para(
        doc,
        "Contrast: continuous cloud LLM calls for every SMS across a growing user base would be a recurring OpEx line. "
        "On-device inference after download is effectively free per message at the API layer.",
    )

    # 9. Open source
    add_heading(doc, "9. What we open-sourced (and why)", 1)
    add_para(
        doc,
        "Open-sourcing builds trust, invites bank/locale contributions, and lets researchers reproduce the gates. "
        "Because training data is synthetic and the base model is Apache 2.0, redistribution is clean.",
    )
    add_table(
        doc,
        ["Artifact", "URL"],
        [
            ["Model (bf16 + adapter)", "https://huggingface.co/finndot/finnai-slm-v2"],
            ["Dataset", "https://huggingface.co/datasets/finndot/finnai-slm-data"],
            ["Code + docs toolkit", "https://github.com/arunbhatg/finnai-slm"],
            ["Product app", "https://github.com/devaka207/Finndot"],
        ],
    )

    # 10. Risks
    add_heading(doc, "10. Limitations & residual risks", 1)
    add_para(
        doc,
        "• Optimized for Indian banking SMS — not a general global SMS model.\n"
        "• Merchant strings remain the hardest field; regex still wins on supported banks.\n"
        "• Chat groundedness at 68% is good vs base but not perfect — coaching can still paraphrase figures.\n"
        "• On-device gate (TTFT / memory on mid-range phones) must be measured on hardware before calling INT4 “fully shipped”.\n"
        "• Templates cannot cover every NPCI variant overnight; monitoring unrecognized SMS still matters.",
    )

    # 11. Ask of leadership
    add_heading(doc, "11. What good looks like next", 1)
    add_para(
        doc,
        "1. Complete on-device INT4 bench and point the app MODEL_URL at CloudFront when gate 5 passes.\n"
        "2. Keep expanding Indic + long-tail banks via synthetic templates (not user dumps).\n"
        "3. Invite community PRs on the public toolkit for new locales.\n"
        "4. Reuse the same playbook for adjacent on-device extraction jobs (receipts, invoices) where privacy matters.",
    )

    add_heading(doc, "Appendix A — Glossary", 1)
    add_table(
        doc,
        ["Term", "Plain meaning"],
        [
            ["LLM", "Large Language Model — software that predicts text (ChatGPT-class systems)"],
            ["SLM", "Small Language Model — same idea, small enough for a phone"],
            ["Fine-tuning", "Extra training so a base model specializes for your task"],
            ["QLoRA", "Efficient fine-tuning: freeze big model, train tiny adapter in 4-bit"],
            ["SFT", "Supervised Fine-Tuning — learn from input→correct output pairs"],
            ["R-EM", "Strict Record Exact-Match — all key SMS fields correct"],
            ["Groundedness", "Every number in the answer appears in the provided ledger"],
            ["INT4 / LiteRT-LM", "Compressed on-device format used by the Android runtime"],
        ],
    )

    add_heading(doc, "Appendix B — Technical hyperparameter snapshot", 1)
    add_para(
        doc,
        "Base Qwen/Qwen3-1.7B; QLoRA NF4 double-quant bf16 compute; LoRA r=16 α=32 dropout 0.05; "
        "targets q/k/v/o/gate/up/down_proj; 3 epochs; lr 2e-4 cosine 3% warmup; batch 2× accum 8; "
        "max len 1536; packing off; seed 42; greedy decode at eval; thinking off.",
        size=10,
    )

    out = OUT_DIR / "FinnAI_SLM_CXO_FineTuning_Brief.docx"
    doc.save(out)
    return out


def main():
    paths = {
        "pipeline": fig_pipeline(),
        "decision": fig_decision(),
        "qlora": fig_qlora(),
        "architecture": fig_architecture(),
        "data_mix": fig_data_mix(),
        "results": fig_results(),
        "privacy": fig_privacy(),
    }
    out = build_doc(paths)
    print(out)


if __name__ == "__main__":
    main()
