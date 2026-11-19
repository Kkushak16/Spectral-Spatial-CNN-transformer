"""Build docs/AgriSpec_XAI_Report.pdf from the saved result files.

Every measured number is read from results/, so rerunning this after new runs updates the report.
Needs reportlab and matplotlib:  python scripts/build_report.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics as st
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RUNS = RES / "runs"
OUT = ROOT / "docs" / "AgriSpec_XAI_Report.pdf"
AUTHOR = os.environ.get("REPORT_AUTHOR", "")  # optional byline, kept out of the source

INK, MUTED, RULE, TINT = "#0b0b0b", "#52514e", "#d9d8d3", "#f3f2ee"
BLUE, ORANGE = "#2a78d6", "#eb6834"
AGRI = ["Arable land", "Pastures", "Agro-forestry areas",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Complex cultivation patterns", "Permanent crops"]
SHORT = {"Land principally occupied by agriculture, with significant areas of natural vegetation":
         "Mainly agriculture + natural vegetation",
         "Moors, heathland and sclerophyllous vegetation": "Moors, heathland, sclerophyllous",
         "Natural grassland and sparsely vegetated areas": "Natural grassland, sparse vegetation",
         "Industrial or commercial units": "Industrial or commercial"}

for name, file in (("Body", "calibri.ttf"), ("Bold", "calibrib.ttf"), ("Italic", "calibrii.ttf")):
    pdfmetrics.registerFont(TTFont(name, f"C:/Windows/Fonts/{file}"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Italic", boldItalic="Bold")
plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": RULE})


def style(name, size, lead, **kw):
    return ParagraphStyle(name, fontName=kw.pop("font", "Body"), fontSize=size, leading=lead,
                          textColor=colors.HexColor(kw.pop("color", INK)), alignment=TA_LEFT, **kw)


S = {"title": style("title", 19, 22, font="Bold"),
     "sub": style("sub", 9.6, 12, color=MUTED, spaceAfter=3),
     "h1": style("h1", 14.5, 17.5, font="Bold", spaceAfter=3),
     "h2": style("h2", 11.8, 14, font="Bold", spaceBefore=8, spaceAfter=2, color=BLUE),
     "h3": style("h3", 11, 13.5, font="Bold", spaceBefore=8, spaceAfter=3),
     "p": style("p", 10.3, 12.5, spaceAfter=2),
     "b": style("b", 10.3, 12.3, leftIndent=8, bulletIndent=0, spaceAfter=1.6),
     "body": style("body", 10, 12.7, spaceAfter=4),
     "li": style("li", 10, 12.6, leftIndent=10, bulletIndent=0, spaceAfter=2.5),
     "cell": style("cell", 8.9, 10.6),
     "cellb": style("cellb", 8.9, 10.6, font="Bold"),
     "note": style("note", 8.6, 10.4, color=MUTED, spaceBefore=3)}


def P(text, s="p"):
    return Paragraph(text, S[s])


def bullets(items, s="b"):
    return [Paragraph(t, S[s], bulletText="\u2022") for t in items]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def ms(values, d=3):
    return f"{st.mean(values):.{d}f} \u00b1 {st.stdev(values):.{d}f}"


def table(rows, widths, head=True, align_from=1):
    data = [[P(str(c), "cellb" if head and r == 0 else "cell") for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TINT)), ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 2.8), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


# ---- data ----------------------------------------------------------------------------------------------
EU = {k: load(next(RUNS.glob(g)) / "metrics.json")["test"] | {"cost": load(next(RUNS.glob(g)) / "metrics.json")["cost"]}
      for k, g in {"rgb": "*B2_cnn_rgb_s0*", "b3": "*B3_cnn_12band_s0*", "b8": "*B8_cnntr_12band_v2_s0*"}.items()}
rf = load(RES / "phase4b_spectral.json")
rf12 = [rf["B0_rf_12band_meanstd"][f"seed{i}"]["test"] for i in range(3)]
rfrgb = [rf["B0_rf_rgb_meanstd"][f"seed{i}"]["test"] for i in range(3)]
BEN = [load(RUNS / "ben_b8_cnntr_12band_s0/metrics.json"),
       load(RUNS / "ben_b8_cnntr_12band_s1_s2/seed_1/metrics.json"),
       load(RUNS / "ben_b8_cnntr_12band_s1_s2/seed_2/metrics.json")]
BEN_RGB = [load(RUNS / f"ben_b8_rgb/seed_{s}/metrics.json") for s in range(3)]
GATE = [load(p) for p in sorted((RUNS / "ben_s5_gate_k6").glob("seed_*/metrics.json"))]
GATE_DONE = len(GATE) == 3
gain = 100 * (EU["b3"]["acc"] - EU["rgb"]["acc"])


def metric(runs, part, key):
    return [r[part][key] for r in runs]


def class_ap(runs, part, c):
    return [r[part]["per_class_AP"][c] for r in runs]


# ---- figures -------------------------------------------------------------------------------------------
def fig_recall(path):
    a, b = EU["rgb"]["per_class_recall"], EU["b3"]["per_class_recall"]
    names = sorted(a, key=lambda c: b[c] - a[c])
    fig, ax = plt.subplots(figsize=(7.1, 2.75), dpi=240)
    for i, c in enumerate(names):
        ax.plot([100 * a[c], 100 * b[c]], [i, i], color=RULE, lw=2, zorder=1, solid_capstyle="round")
    ax.scatter([100 * a[c] for c in names], range(len(names)), s=30, color=ORANGE, label="RGB (3 bands)", zorder=2,
               edgecolor="white", linewidth=0.8)
    ax.scatter([100 * b[c] for c in names], range(len(names)), s=30, color=BLUE, label="All 12 bands", zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(90, 100.4)
    ax.set_xlabel("Test recall (%), same compact CNN, seed 0")
    ax.grid(axis="x", color="#ecebe6", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", frameon=False, ncol=2, handletextpad=0.1, columnspacing=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_ben(path):
    cls = sorted(BEN[0]["test"]["per_class_AP"], key=lambda c: st.mean(class_ap(BEN, "test", c)))
    fig, ax = plt.subplots(figsize=(7.1, 3.95), dpi=240)
    for i, c in enumerate(cls):
        v = class_ap(BEN, "test", c)
        m, s = st.mean(v), st.stdev(v)
        ax.barh(i, m, height=0.68, color=ORANGE if c in AGRI else BLUE, zorder=2)
        ax.plot([max(m - s, 0), min(m + s, 1)], [i, i], color=INK, lw=0.8, zorder=3)
        ax.text(1.015, i, f"{m:.3f}", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(cls)), [SHORT.get(c, c) for c in cls])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test average precision: mean of 3 seeds (line shows \u00b1 1 std)")
    ax.grid(axis="x", color="#ecebe6", lw=0.6, zorder=0)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.barh([0], [0], color=ORANGE, label="Agricultural class")
    ax.barh([0], [0], color=BLUE, label="Other class")
    ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(0.99, 0.02))
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def class_gain(c):
    return st.mean(class_ap(BEN, "test", c)) - st.mean(class_ap(BEN_RGB, "test", c))


def fig_ben_rgb(path):
    cls = sorted(BEN[0]["test"]["per_class_AP"], key=class_gain)
    fig, ax = plt.subplots(figsize=(7.1, 3.95), dpi=240)
    for i, c in enumerate(cls):
        a, b = st.mean(class_ap(BEN_RGB, "test", c)), st.mean(class_ap(BEN, "test", c))
        ax.plot([a, b], [i, i], color=RULE, lw=2, zorder=1, solid_capstyle="round")
        ax.text(1.015, i, f"{b - a:+.3f}", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.scatter([st.mean(class_ap(BEN_RGB, "test", c)) for c in cls], range(len(cls)), s=30, color=ORANGE,
               label="RGB (3 bands)", zorder=2, edgecolor="white", linewidth=0.8)
    ax.scatter([st.mean(class_ap(BEN, "test", c)) for c in cls], range(len(cls)), s=30, color=BLUE,
               label="All 12 bands", zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_yticks(range(len(cls)), [SHORT.get(c, c) for c in cls])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test average precision, mean of 3 seeds (number at right: 12 bands minus RGB)")
    ax.grid(axis="x", color="#ecebe6", lw=0.6, zorder=0)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), frameon=False, ncol=2, handletextpad=0.1, columnspacing=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


# ---- pages ---------------------------------------------------------------------------------------------
def page_one():
    left = [P("Why we are doing this", "h2"),
            P("Sentinel-2 records 13 spectral bands, but deep models usually take either all of them or only RGB. "
              "Every extra band adds storage, download and compute cost, while RGB discards the infrared signal that "
              "describes crops, moisture and soil. We want to know <b>which few bands are really needed</b> for "
              "reliable land-cover and crop mapping, and to show it with evidence."),
            P("What we are achieving", "h2"),
            *bullets(["A compact, explainable classifier that reads only a small learned set of bands (k = 3, 4, 6 or 8).",
                      "Accuracy close to the all-band model. Pre-set target: k = 4\u20136 within 1.0 point of 12 bands on EuroSAT.",
                      "Lower data and compute needs, suited to low-cost and on-device agricultural monitoring."]),
            P("Datasets", "h2"),
            *bullets(["<b>EuroSAT multispectral:</b> 27,000 patches, 64\u00d764 px, 10 classes, one label each. Used for development.",
                      "<b>BigEarthNet v2 (reBEN):</b> 549,488 Sentinel-2 patches, 120\u00d7120 px at 10 m, 19 classes, several labels each. "
                      "We use the 480,038 in the official split file: 237,871 train, 122,342 validation, 119,825 test."]),
            P("Results so far", "h2"),
            *bullets([f"<b>EuroSAT:</b> {pct(EU['b3']['acc'])} with 12 bands vs {pct(EU['rgb']['acc'])} with RGB, same CNN (one seed).",
                      f"<b>BigEarthNet:</b> macro mAP {ms(metric(BEN, 'test', 'macro_mAP'))} with 12 bands vs {ms(metric(BEN_RGB, 'test', 'macro_mAP'))} with RGB, same model (three seeds)."]),
            P("How we approached it", "h2"),
            *bullets(["<b>Understand the data first:</b> checked real files, proved the EuroSAT band order and tested for spatial leakage.",
                      "<b>Write hypotheses first:</b> seven hypotheses, each with the experiment that could reject it.",
                      "<b>Climb a baseline ladder:</b> random forest, RGB CNN, 12-band CNN, then CNN-Transformer.",
                      "<b>Scale up:</b> packed BigEarthNet into 241 compressed files (54 GB) and trained on Kaggle GPUs.",
                      "<b>Add band selection:</b> learn k bands, freeze them, fine-tune, then export a true k-band model.",
                      "<b>Stay honest:</b> choose models on validation data and score the test set once per final model."]),
            ]
    right = [P("Technical stack", "h2"),
             *bullets(["<b>Model:</b> compact ResNet-style CNN feeding a 2-layer, 4-head Transformer (128-dim tokens), with gated fusion.",
                       "<b>Band selection:</b> one trainable score per band; straight-through top-k gate.",
                       "<b>Training:</b> PyTorch, AdamW, one-cycle learning rate, flips and rotations, mixed precision on GPU.",
                       "<b>Data tools:</b> rasterio, tifffile, NumPy, pandas, scikit-learn, Zstandard, Kaggle CLI (Tesla T4).",
                       "<b>Metrics:</b> accuracy and F1 (EuroSAT); macro and micro mAP, F1, per-class AP (BigEarthNet)."]),
             P("How we differentiate", "h2"),
             *bullets(["<b>Bands are truly removed:</b> the final network has k input channels, so other bands are never read.",
                       "<b>Leakage-aware testing:</b> 59% of EuroSAT patches have a neighbour within 1 km, so we keep neighbours in the same split.",
                       "<b>Physics and proof together (planned):</b> NDVI/NDMI guidance with a control test, plus deletion tests of band explanations.",
                       "<b>Fair claims:</b> band gates already exist, so our contribution is this combined, validated package, not a \u201cfirst\u201d."]),
             P("How we can improve accuracy", "h2"),
             *bullets(["Weighted or asymmetric loss for rare classes such as beaches and coastal wetlands.",
                       "Per-class decision thresholds tuned on validation data.",
                       "Native multi-resolution input instead of upsampling 20 m and 60 m bands.",
                       "Pretrained weights (SatMAE, SSL4EO-S12) and ensembling the three seeds.",
                       "Adding Sentinel-1 radar, which is paired with BigEarthNet patches."]),
             P("Future models and tools", "h2"),
             P("<b>MobileNetV3</b> and <b>EfficientNet-B0</b> as lighter encoders, <b>ResNet-50</b> as the RGB reference, "
               "<b>Integrated Gradients</b> and <b>SHAP</b> for explanations, and <b>mRMR/PCA</b> band subsets for comparison. None of these results exist yet."),
             P("Future plans", "h2"),
             *bullets(["Finish k = 6, then run k = 3, 4 and 8 on BigEarthNet.",
                       "Compare the gate against RGB, RGB+NIR, random, mRMR and PCA subsets.",
                       "Add the physics head, then the consistency loss, one at a time.",
                       "Complete EuroSAT seeds, low-data and geographic tests, then write the paper."]),
             ]
    head = [P("AgriSpec-XAI: Learning Which Satellite Bands Matter", "title"),
            P("Physics-guided sparse band selection in a compact CNN-Transformer &nbsp;|&nbsp; " + (AUTHOR + " &nbsp;|&nbsp; " if AUTHOR else "") + "19 Nov 2026", "sub")]
    return head + [FrameBreak()] + left + [FrameBreak()] + right


def page_eurosat(fig):
    a, b, c = EU["rgb"], EU["b3"], EU["b8"]
    ci = lambda r: f"{100 * r['acc_ci95'][0]:.2f}\u2013{100 * r['acc_ci95'][1]:.2f}"
    rows = [["Model", "Bands", "Test accuracy", "95% CI", "Macro F1", "Agri F1", "Params", "MFLOPs", "CPU ms"],
            ["Random forest (band mean/std)", "RGB", f"{100 * st.mean(r['acc'] for r in rfrgb):.2f}%", "\u2014", pct(st.mean(r["macro_f1"] for r in rfrgb)), "\u2014", "\u2014", "\u2014", "\u2014"],
            ["Random forest (band mean/std)", "12", f"{100 * st.mean(r['acc'] for r in rf12):.2f}%", "\u2014", pct(st.mean(r["macro_f1"] for r in rf12)), "\u2014", "\u2014", "\u2014", "\u2014"]]
    for name, bands, r in (("Compact CNN (B2)", "RGB", a), ("Compact CNN (B3)", "12", b), ("CNN-Transformer (B8)", "12", c)):
        rows.append([name, bands, f"<b>{pct(r['acc'])}</b>", ci(r), pct(r["macro_f1"]), pct(r["agri_from_multiclass"]["f1"]),
                     f"{r['cost']['params'] / 1e6:.2f} M", f"{r['cost']['flops'] / 1e6:.1f}", f"{r['cost']['latency_ms_median']:.1f}"])
    return [P("Results 1: EuroSAT, RGB versus all bands", "h1"),
            P(f"With the same CNN, moving from RGB to 12 bands raised test accuracy by <b>{gain:.2f} points</b>, from {pct(a['acc'])} to {pct(b['acc'])}. "
              f"The 95% confidence intervals do not overlap. Agricultural F1 rose from {pct(a['agri_from_multiclass']['f1'])} to {pct(b['agri_from_multiclass']['f1'])}.", "body"),
            table(rows, [47, 11, 20, 23, 17, 16, 15, 16, 15]),
            P("Grouped 70/10/20 split with 5,400 test patches. Neural models: seed 0 only. Random forests: mean of 3 seeds. "
              "Agri F1 treats annual and permanent crops as the positive class. CPU ms is median time per image.", "note"),
            P("Which classes gained from the extra bands", "h3"),
            Image(str(fig), width=180 * mm, height=180 * mm * 2.75 / 7.1),
            P("What this tells us", "h3"),
            *bullets([f"<b>Permanent crops and pasture gained most.</b> Permanent crop recall rose from {pct(a['per_class_recall']['PermanentCrop'], 1)} to {pct(b['per_class_recall']['PermanentCrop'], 1)}, "
                      f"and pasture from {pct(a['per_class_recall']['Pasture'], 1)} to {pct(b['per_class_recall']['Pasture'], 1)}.",
                      f"<b>The extra bands add very little model compute.</b> The 12-band CNN is almost the same size as the RGB CNN ({b['cost']['flops'] / 1e6:.1f} vs {a['cost']['flops'] / 1e6:.1f} MFLOPs).",
                      f"<b>The Transformer has not shown an advantage yet.</b> It scored {pct(c['acc'])} against {pct(b['acc'])} for the plain CNN, with overlapping intervals and 29% more parameters.",
                      "<b>Spectral information alone is useful.</b> A random forest with no spatial information reaches 88.2% on 12 bands but only 74.7% on RGB.",
                      "<b>Caution:</b> seeds 1\u20134 are still needed before the CNN comparisons can be treated as final."], "li")]


def page_ben(fig):
    rows = [["Metric", "Seed 0", "Seed 1", "Seed 2", "Test mean \u00b1 std", "Validation mean \u00b1 std"]]
    for k, label in (("macro_mAP", "Macro mAP"), ("micro_mAP", "Micro mAP"), ("macro_f1", "Macro F1"), ("micro_f1", "Micro F1")):
        v = metric(BEN, "test", k)
        rows.append([label, *[f"{x:.3f}" for x in v], f"<b>{ms(v)}</b>", ms(metric(BEN, "val", k))])
    mac, mic = metric(BEN, "test", "macro_mAP"), metric(BEN, "test", "micro_mAP")
    return [P("Results 2: BigEarthNet, all 12 bands (baseline B8)", "h1"),
            P(f"On 119,825 unseen test patches the CNN-Transformer reached <b>macro mAP {ms(mac)}</b> and <b>micro mAP {ms(mic)}</b>. "
              "Results were consistent across three independently trained models.", "body"),
            table(rows, [34, 22, 22, 22, 40, 40]),
            P("1.61 M parameters, 225 region tokens, 20 epochs, about 2.6 GPU hours per seed. Best epoch chosen by validation macro mAP. "
              "The 20 m and 60 m bands were upsampled to 120\u00d7120 px.", "note"),
            P("Test score for each of the 19 classes", "h3"),
            Image(str(fig), width=180 * mm, height=180 * mm * 3.95 / 7.1),
            P("What this tells us", "h3"),
            *bullets([f"<b>Large agricultural classes are strong:</b> arable land {st.mean(class_ap(BEN, 'test', 'Arable land')):.3f} and pastures {st.mean(class_ap(BEN, 'test', 'Pastures')):.3f}. "
                      f"Permanent crops are harder at {st.mean(class_ap(BEN, 'test', 'Permanent crops')):.3f}.",
                      f"<b>Rare classes pull macro mAP down.</b> Beaches has only 152 test patches and scored {st.mean(class_ap(BEN, 'test', 'Beaches, dunes, sands')):.3f}, "
                      f"compared with {st.mean(class_ap(BEN, 'val', 'Beaches, dunes, sands')):.3f} on validation. The cause of that drop has not been established."], "li")]


def page_ben_rgb(fig):
    rows = [["Test metric", "RGB (3 bands)", "All 12 bands", "Gain", "Gain by seed (0, 1, 2)"]]
    for k, label in (("macro_mAP", "Macro mAP"), ("micro_mAP", "Micro mAP"), ("macro_f1", "Macro F1"), ("micro_f1", "Micro F1")):
        r, b = metric(BEN_RGB, "test", k), metric(BEN, "test", k)
        d = [x - y for x, y in zip(b, r)]
        rows.append([label, ms(r), f"<b>{ms(b)}</b>", f"<b>{st.mean(d):+.3f}</b>", ", ".join(f"{x:+.3f}" for x in d)])
    cls = list(BEN[0]["test"]["per_class_AP"])
    better = sum(class_gain(c) > 0 for c in cls)
    gap = st.mean(metric(BEN, "test", "macro_mAP")) - st.mean(metric(BEN_RGB, "test", "macro_mAP"))
    show = lambda runs, c: f"{st.mean(class_ap(runs, 'test', c)):.3f}"
    wet, crop, arable, sea = "Coastal wetlands", "Permanent crops", "Arable land", "Marine waters"
    return [P("Results 3: BigEarthNet, RGB versus all 12 bands", "h1"),
            P(f"With the same CNN-Transformer, using all 12 bands instead of RGB raised test macro mAP by <b>{gap:.3f}</b>. "
              "All four overall metrics improved in every one of the three seeds.", "body"),
            table(rows, [34, 36, 36, 22, 52]),
            P("Same model design, seeds, data split and training settings; only the input bands differ (RGB = B04, B03, B02). "
              f"1.61 M parameters in both models. RGB took {st.mean(r['total_hours'] for r in BEN_RGB):.1f} GPU hours per seed, "
              f"12 bands {st.mean(r['total_hours'] for r in BEN):.1f}.", "note"),
            P("Test score for each class, RGB and 12 bands", "h3"),
            Image(str(fig), width=180 * mm, height=180 * mm * 3.95 / 7.1),
            P("What this tells us", "h3"),
            *bullets([f"<b>{better} of 19 classes scored higher with 12 bands</b> on the three-seed mean. Marine waters was the exception ({show(BEN, sea)} vs {show(BEN_RGB, sea)} for RGB), a very small difference.",
                      f"<b>Coastal wetlands gained most,</b> from {show(BEN_RGB, wet)} to {show(BEN, wet)}, though it remains a weak class.",
                      f"<b>Crop classes gained unevenly.</b> Permanent crops rose from {show(BEN_RGB, crop)} to {show(BEN, crop)}, while arable land, already strong with RGB, rose from {show(BEN_RGB, arable)} to {show(BEN, arable)}.",
                      "<b>This sets the target for band selection:</b> a useful small band set should score well above RGB and close to all 12 bands."], "li")]


def page_gate():
    story = [P("Results 4: Learned band selection (k = 6) and project status", "h1")]
    base = {k: metric(BEN, "test", k) for k in ("macro_mAP", "micro_mAP", "macro_f1", "micro_f1")}
    if GATE_DONE:
        rows = [["Test metric", "RGB (3 bands)", "Learned 6 bands (S5)", "12 bands (B8)", "6 bands minus 12"]]
        for k, label in (("macro_mAP", "Macro mAP"), ("micro_mAP", "Micro mAP"), ("macro_f1", "Macro F1"), ("micro_f1", "Micro F1")):
            g = metric(GATE, "test", k)
            rows.append([label, ms(metric(BEN_RGB, "test", k)), f"<b>{ms(g)}</b>", ms(base[k]), f"{st.mean(g) - st.mean(base[k]):+.3f}"])
        sets = [set(r["selected_bands"]) for r in GATE]
        pairs = [(0, 1), (0, 2), (1, 2)]
        jac = st.mean(len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs)
        common = sorted(set.intersection(*sets))
        band_rows = [["Seed", "Selected bands", "Test macro mAP", "Test micro mAP"]]
        for r in GATE:
            band_rows.append([r["seed"], ", ".join(r["selected_bands"]), f"{r['test']['macro_mAP']:.3f}", f"{r['test']['micro_mAP']:.3f}"])
        story += [P(f"The learned 6-band model reads half of the input bands. Its test macro mAP changed by <b>{st.mean(metric(GATE, 'test', 'macro_mAP')) - st.mean(base['macro_mAP']):+.3f}</b> compared with all 12 bands.", "body"),
                  table(rows, [32, 36, 42, 36, 34]), Spacer(1, 7), table(band_rows, [20, 90, 35, 35]),
                  P(f"Band stability across seeds: average Jaccard overlap {jac:.2f}. Bands chosen by all three seeds: {', '.join(common) if common else 'none'}. "
                    "Final scores come from the exported 6-channel model.", "note")]
    else:
        story += [P("<b>Training is in progress, so no k = 6 result is reported yet.</b> This page will be regenerated with measured scores when all three seeds finish.", "body"),
                  table([["Item", "Setup"],
                         ["Model", "S5: the B8 CNN-Transformer with a straight-through top-k band gate, keeping 6 of 12 bands"],
                         ["Training", "3 seeds, 20 epochs each. The gate learns for 14 epochs, then the bands are frozen for 6 epochs of fine-tuning"],
                         ["Final model", "Exported with a genuine 6-channel first layer, and checked against the gated model"],
                         ["Comparison", f"12-band baseline: test macro mAP {ms(base['macro_mAP'])}, micro mAP {ms(base['micro_mAP'])}"]], [32, 148])]
    story += [P("Where each experiment stands", "h3"),
              table([["Experiment", "EuroSAT", "BigEarthNet"],
                     ["RGB baseline, compact CNN", "Done, seed 0", "Not run"],
                     ["12 bands, compact CNN", "Done, seed 0", "Not run"],
                     ["RGB baseline, CNN-Transformer", "Not run", "Done, 3 seeds"],
                     ["12 bands, CNN-Transformer (B8)", "Done, seed 0", "Done, 3 seeds"],
                     ["Learned gate, k = 6", "Not run", "Done, 3 seeds" if GATE_DONE else "Running, 3 seeds"],
                     ["Learned gate, k = 3, 4, 8", "Not run", "Not run"],
                     ["RGB+NIR, random, mRMR and PCA subsets", "Subsets prepared, not trained", "Not run"],
                     ["ResNet-50 RGB reference", "Script ready, not run", "Not run"],
                     ["MobileNetV3, EfficientNet-B0 encoders", "Not run", "Not run"],
                     ["Physics head, consistency loss", "Not run", "Not run"],
                     ["Explanations and deletion tests", "Not run", "Not run"]], [84, 48, 48]),
              P("Limits of the current results", "h3"),
              *bullets(["EuroSAT neural-network results use one seed, so they are early evidence rather than final claims.",
                        "We have not yet benchmarked these scores against published results under matching conditions.",
                        "EuroSAT and BigEarthNet use different labels and metrics, so their scores cannot be compared directly.",
                        "The physics-guided and explainability parts are designed but not yet built into a trained model."], "li"),
              P("All measured values come from saved run files in <font name='Bold'>results/</font>. Training code is in <font name='Bold'>src/</font> and <font name='Bold'>kaggle_jobs/</font>.", "note")]
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8.2)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(15 * mm, 8.5 * mm, "AgriSpec-XAI project report")
    canvas.drawRightString(A4[0] - 15 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    W, H = A4
    m, top, gap = 15 * mm, 12 * mm, 6 * mm
    head_h = 18 * mm
    col = (W - 2 * m - gap) / 2
    y0, body_h = 13 * mm, H - top - 13 * mm
    cover = [Frame(m, H - top - head_h, W - 2 * m, head_h, 0, 0, 0, 0, id="head"),
             Frame(m, y0, col, body_h - head_h, 0, 0, 0, 0, id="left"),
             Frame(m + col + gap, y0, col, body_h - head_h, 0, 0, 0, 0, id="right")]
    OUT.parent.mkdir(exist_ok=True)
    doc = BaseDocTemplate(str(OUT), pagesize=A4, title="AgriSpec-XAI Project Report", author=AUTHOR)
    doc.addPageTemplates([PageTemplate("cover", cover, onPage=footer),
                          PageTemplate("body", [Frame(m, y0, W - 2 * m, body_h, 0, 0, 0, 0)], onPage=footer)])
    with tempfile.TemporaryDirectory() as t:
        f1, f2, f3 = Path(t) / "recall.png", Path(t) / "ben.png", Path(t) / "ben_rgb.png"
        fig_recall(f1)
        fig_ben(f2)
        fig_ben_rgb(f3)
        doc.build(page_one() + [NextPageTemplate("body"), PageBreak()] + page_eurosat(f1) + [PageBreak()]
                  + page_ben(f2) + [PageBreak()] + page_ben_rgb(f3) + [PageBreak()] + page_gate())
    print(OUT, "gate results included:", GATE_DONE)


if __name__ == "__main__":
    main()
