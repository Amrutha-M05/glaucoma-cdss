"""Write the architecture-variant tables as a compilable LaTeX document (booktabs, journal style).

Numbers in Table 2 are read from the run folders, so the .tex always matches the results.
Usage: python figures/make_latex_tables.py  ->  figures/architecture_variants.tex
Compile with pdflatex (e.g. Overleaf, MiKTeX or TeX Live). The three table environments can be
copied into a manuscript unchanged (needs booktabs, threeparttable, amssymb, pifont).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdss.paths import RUNS_DIR  # noqa: E402

OUT = ROOT / "figures" / "architecture_variants.tex"


def load(path):
    p = RUNS_DIR / path
    return json.loads(p.read_text()) if p.exists() else None


def run_status(tag):
    if (RUNS_DIR / "p2" / f"BTViT_{tag}" / "seed2026" / "metrics.json").exists():
        return "Completed"
    log = ROOT / "runs" / "logs" / f"p2_{tag}.log"
    return "Running" if log.exists() and "epoch" in log.read_text(errors="ignore") else "Queued"


def results_rows():
    models = [("V0: Pure ViT", load("p2/PureViT_purevit_tiny/seed2026/metrics.json"), "5.5"),
              ("V1: BT-ViT v1", load("p2/BTViT_full/seed2026/metrics_tta.json"), "4.9"),
              ("V2: BT-ViT v2", load("p2/BTViT_v2/seed2026/metrics.json"), "6.7"),
              ("B1: Nasrin et al.~\\cite{nasrin2026}", None, "10.6")]
    spec = [("test", "auc"), ("test", "accuracy"), ("test", "sensitivity"), ("test", "specificity"),
            ("RIM-ONE", "auc"), ("RIM-ONE", "accuracy"), ("DRISHTI-GS", "auc"), ("DRISHTI-GS", "accuracy")]

    def val(m, g, k):
        if m is None:
            return None
        return round((m["metrics"] if k == "auc" else m["metrics_accthr"])[g][k], 3)
    values = [[val(m, g, k) for g, k in spec] for _, m, _ in models]
    best = [max(v[i] for v in values if v[i] is not None) for i in range(len(spec))]
    lines = []
    for (name, _, params), vals in zip(models, values):
        cells = []
        for i, v in enumerate(vals):
            if v is None:
                cells.append("--")
            else:
                s = f"{v:.3f}"
                cells.append(f"\\textbf{{{s}}}" if v == best[i] else s)
        lines.append(f"{name} & {params} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def experiments_rows():
    """V2 training-recipe variants: modification, validation AUC (selection criterion), test/external AUC."""
    exps = [("v2", "V2 (reference)", "--"),
            ("v2_ema", "V2 + EMA", "EMA of weights (decay 0.999)"),
            ("v2_ls01", "V2 + label smoothing", r"Smoothed targets ($\varepsilon = 0.1$)"),
            ("v2_bio1", r"V2 + $\lambda_{\mathrm{bio}} = 1.0$", "Doubled biomarker supervision"),
            ("v2_opengates", "V2 + open gates", r"No weight decay on gates; $\beta_0 = -2$")]
    groups = ["val", "test", "RIM-ONE", "DRISHTI-GS"]
    metrics = {tag: load(f"p2/BTViT_{tag}/seed2026/metrics.json") for tag, _, _ in exps}
    done = [m for m in metrics.values() if m is not None]
    best = {g: max(round(m["metrics"][g]["auc"], 3) for m in done) for g in groups}
    lines = []
    for tag, name, mod in exps:
        m = metrics[tag]
        if m is None:
            cells = ["--"] * len(groups) + [run_status(tag)]
        else:
            cells = []
            for g in groups:
                v = round(m["metrics"][g]["auc"], 3)
                cells.append(f"\\textbf{{{v:.3f}}}" if v == best[g] else f"{v:.3f}")
            cells.append("Completed")
        lines.append(f"{name} & {mod} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def v1_layer_rows():
    """Layer-by-layer table of BT-ViT v1 with parameter counts taken from the model itself."""
    from cdss.models.btvit import BTViT
    m = BTViT(pretrained=False)
    P = lambda mod: sum(p.numel() for p in mod.parameters())  # noqa: E731
    f = lambda n: f"{n:,}".replace(",", "{,}")  # noqa: E731
    conv = r"[Conv$3{\times}3$, BN, GELU]$\times$2"
    rows = [
        ("Input", "--", r"Disc-centred ROI, RGB + CLAHE", r"$3{\times}224{\times}224$", "--"),
        ("Conv stem", "Block 1", conv + r", 3$\to$32, stride 2", r"$32{\times}112{\times}112$", f(P(m.stem[0]))),
        ("", "Block 2", conv + r", 32$\to$64, stride 2", r"$64{\times}56{\times}56$", f(P(m.stem[1]))),
        ("", "Block 3", conv + r", 64$\to$128, stride 2", r"$128{\times}28{\times}28$", f(P(m.stem[2]))),
        ("Patch projection", "--", r"Conv$2{\times}2$, stride 2, 128$\to$192", r"$196{\times}192$", f(P(m.proj))),
        ("Token assembly", "--", r"[CLS] + learned position embedding$^{\dagger}$", r"$197{\times}192$",
         f(m.cls_token.numel() + m.pos_embed.numel())),
        ("Transformer", r"Blocks 1--4$^{\dagger}$", r"LN, MHSA (3 heads, $d_h{=}64$), LN, MLP 192$\to$768$\to$192",
         r"$197{\times}192$", f(P(m.pre))),
        ("Seg. decoder", "Up 1", r"$\uparrow$2, $\oplus$ stem block 3, ConvBlock 320$\to$128", r"$128{\times}28{\times}28$",
         f(P(m.dec[0]))),
        ("", "Up 2", r"$\uparrow$2, $\oplus$ stem block 2, ConvBlock 192$\to$64", r"$64{\times}56{\times}56$", f(P(m.dec[1]))),
        ("", "Up 3", r"$\uparrow$2, $\oplus$ stem block 1, ConvBlock 96$\to$32", r"$32{\times}112{\times}112$", f(P(m.dec[2]))),
        ("", "Head", r"Conv$1{\times}1$, 32$\to$3, $\uparrow$2, softmax (bg/disc/cup)", r"$3{\times}224{\times}224$",
         f(P(m.seg_head))),
        ("Biomarkers", "--", "12 differentiable biomarkers from soft masks, z-scored", "12", "0"),
        ("Clinical tokens", r"4 MLPs", r"Linear($k\to$192), GELU, Linear(192$\to$192) + type emb.; $k{=}3,5,2,2$",
         r"$4{\times}192$", f(P(m.token_mlps) + m.type_embed.numel())),
        ("Transformer", r"Blocks 5--8$^{\dagger}$", r"as above + rim bias $\alpha_l\log(r_j{+}0.05)$ on CLS$\to$patch",
         r"$201{\times}192$", f(P(m.post) + m.alpha.numel())),
        ("", "Final LN", "LayerNorm", r"$201{\times}192$", f(P(m.norm))),
        ("Classifier", "Head", r"concat(CLS, mean clin.) 384$\to$192, GELU, Dropout 0.2, $\to$2, softmax", "2",
         f(P(m.head))),
    ]
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return body + "\n\\midrule\n" + rf"\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Total}}}} & \textbf{{{f(P(m))}}} \\"


def v1_results_rows(path):
    """Per-set classification results of BT-ViT v1 at both validation-fitted thresholds."""
    m = load(path)
    names = {"val": "Validation", "test": "Internal test", "RIM-ONE": "RIM-ONE (ext.)", "DRISHTI-GS": "DRISHTI-GS (ext.)"}
    lines = []
    for g, label in names.items():
        a, b = m["metrics"][g], m["metrics_accthr"][g]
        (tn, fp), (fn, tp) = a["confusion"]
        lines.append(
            f"{label} & {a['n']} & {a['positives']} & {a['auc']:.3f} & {a['accuracy']:.3f} & {a['sensitivity']:.3f} & "
            f"{a['specificity']:.3f} & {a['f1']:.3f} & {b['accuracy']:.3f} & {b['sensitivity']:.3f} & "
            f"{b['specificity']:.3f} & {b['f1']:.3f} & {a['ece']:.3f} & {tp}/{fn}/{fp}/{tn} \\\\")
    return "\n".join(lines), m["threshold"], m["threshold_acc"]


def v1_seg_rows():
    seg = load("p2/BTViT_full/seed2026/seg_metrics.json")
    names = {"val": "Validation", "test": "Internal test", "RIM-ONE": "RIM-ONE (ext.)", "DRISHTI-GS": "DRISHTI-GS (ext.)"}
    return "\n".join(
        f"{label} & {s['disc_dice']:.3f} & {s['disc_iou']:.3f} & {s['cup_dice']:.3f} & {s['cup_iou']:.3f} & "
        f"{s['vcdr_abs_err']:.3f} & {s['cdr_area_abs_err']:.3f} \\\\"
        for g, label in names.items() for s in [seg[g]])


TEMPLATE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{newtxtext,newtxmath}   % Times text and maths
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{amssymb}               % \checkmark
\usepackage{pifont}                % \ding{55} (cross)
\usepackage{array}
\usepackage[labelfont=bf,labelsep=period,skip=4pt]{caption}
\newcommand{\cmark}{\checkmark}
\newcommand{\xmark}{\ding{55}}
\pagestyle{empty}

\begin{document}

\noindent\textbf{Biomarker-Token Vision Transformer (BT-ViT).}
BT-ViT is the proposed glaucoma classifier. It is a Vision Transformer that, alongside the
classification, segments the optic disc and optic cup within the same network, computes twelve clinical
biomarkers from its own soft segmentation masks (area- and vertical/horizontal cup-to-disc ratios,
neuroretinal-rim fractions by ISNT sector, rim ratio, ISNT-rule violation, disc size and shape, and
disc/cup boundary strength), and feeds them back into the transformer as \emph{clinical tokens} that attend
together with the image patch tokens. A rim-guided attention bias further steers the classification token
towards the neuroretinal rim. Two versions were developed (Table~\ref{tab:variants}). \emph{BT-ViT~v1}
replaces the pretrained patch embedding with a convolutional stem and uses eight pretrained blocks.
\emph{BT-ViT~v2} keeps the complete ImageNet-pretrained DeiT-Tiny backbone---identical to the pure-ViT
baseline (V0)---and adds the segmentation branch after block~4 and gated clinical tokens with rim-guided
attention in blocks~9--12; all gates are initialised closed, so v2 starts as the baseline and departs from
it only where the additions reduce the training loss.

\vspace{1em}

%% ---------------------------------------------------------------------------------------- Table 1
\begin{table}[h]
\centering
\begin{threeparttable}
\caption{Architectural comparison of the evaluated variants.}
\label{tab:variants}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Component & V0: Pure ViT & V1: BT-ViT v1 & V2: BT-ViT v2 & B1: Nasrin et al. \\
\midrule
Backbone$^{a}$            & DeiT-Tiny (12)   & DeiT-Tiny (8)    & DeiT-Tiny (12)     & ViT, scratch (6) \\
Patch embedding           & Pretrained $16{\times}16$ & New conv stem & Pretrained $16{\times}16$ & Linear $8{\times}8$ \\
Image tokens              & 196              & 196              & 196                & 784 \\
Disc/cup segmentation     & \xmark           & \cmark\ (internal) & \cmark\ (internal) & \cmark\ (separate CNN) \\
Clinical biomarkers       & \xmark           & \cmark\ (12)     & \cmark\ (12)       & \cmark\ (7) \\
Biomarker fusion          & --               & Tokens, blocks 5--8 & Gated tokens, blocks 9--12 & Late concatenation \\
Rim-guided attention      & \xmark           & \cmark           & \cmark\ (gated)    & \xmark \\
Input                     & RGB + CLAHE      & RGB + CLAHE      & RGB + CLAHE        & Grayscale \\
Parameters (M)            & 5.5              & 4.9              & 6.7                & 10.6 \\
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item[] $^{a}$Number of transformer blocks in parentheses. B1 is our re-implementation of the reference
model~\cite{nasrin2026}; widths not specified in the original are documented in the code.
\item[] Abbreviations: BT-ViT, Biomarker-Token Vision Transformer (proposed); ViT, Vision Transformer;
DeiT, Data-efficient image Transformer~\cite{touvron2021}; CLAHE, contrast-limited adaptive histogram
equalisation; ISNT, inferior--superior--nasal--temporal rim rule; CNN, convolutional neural network.
\end{tablenotes}
\end{threeparttable}
\end{table}

%% ---------------------------------------------------------------------------------------- Table 2
\begin{table}[h]
\centering
\begin{threeparttable}
\caption{Glaucoma classification on the internal test set and two external datasets.}
\label{tab:results}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}lc cccc cc cc@{}}
\toprule
 & & \multicolumn{4}{c}{Internal test} & \multicolumn{2}{c}{RIM-ONE (external)} & \multicolumn{2}{c}{DRISHTI-GS (external)} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-8} \cmidrule(l){9-10}
Model & Params (M) & AUC & Acc. & Sens. & Spec. & AUC & Acc. & AUC & Acc. \\
\midrule
%%RESULTS%%
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item[] All models share data, splits, augmentation, optimiser and schedule (50 epochs, early stopping on
validation AUC) and use horizontal-flip test-time augmentation; single seed. Accuracy, sensitivity and
specificity use the threshold that maximises validation accuracy; no test or external data were used for
any choice. Test sets: $n = 290$ (internal), 485 (RIM-ONE), 101 (DRISHTI-GS); RIM-ONE and DRISHTI-GS were
never seen during training. Best value per column in bold; -- = not yet trained.
\end{tablenotes}
\end{threeparttable}
\end{table}

%% ---------------------------------------------------------------------------------------- Table 3
\begin{table}[h]
\centering
\begin{threeparttable}
\caption{Training-recipe variants of V2 (one modification at a time; AUC).}
\label{tab:experiments}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}ll cccc l@{}}
\toprule
 & & \multicolumn{1}{c}{Selection} & \multicolumn{3}{c}{Evaluation} & \\
\cmidrule(lr){3-3} \cmidrule(lr){4-6}
Run & Modification & Val. & Test & RIM-ONE & DRISHTI-GS & Status \\
\midrule
%%EXPERIMENTS%%
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item[] A variant is adopted only if it improves validation AUC; test and external results are reported but
never used for selection. In V2 the learned gates (rim-attention $\alpha$, clinical-token key bias $\beta$,
classifier gate) remained near their closed initial values, indicating that its gain over V0 stems from
multi-task disc/cup segmentation and biomarker supervision. The selected setting will be compared with V0
over five seeds. Single seed; -- = not yet completed.
\end{tablenotes}
\end{threeparttable}
\end{table}

\clearpage
\noindent\textbf{BT-ViT v1 in detail.} The first BT-ViT design, trained on 2 November 2026 (run
\texttt{BTViT\_full}, seed 2026), is detailed layer by layer in Table~\ref{tab:v1layers}, with its training
configuration in Table~\ref{tab:v1config}, classification results in Table~\ref{tab:v1results} and its own
disc/cup segmentation in Table~\ref{tab:v1seg}.

%% ---------------------------------------------------------------------------------------- Table 4
\begin{table}[h]
\centering
\begin{threeparttable}
\caption{BT-ViT v1: layer-by-layer architecture (batch dimension omitted).}
\label{tab:v1layers}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llp{7.4cm}lr@{}}
\toprule
Stage & Layer & Operation & Output & Params \\
\midrule
%%V1LAYERS%%
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item[] $^{\dagger}$Initialised from ImageNet DeiT-Tiny~\cite{touvron2021} (its first eight of twelve blocks,
[CLS] token and $14{\times}14$ position grid); all other layers are randomly initialised. Each transformer
block has 444{,}864 parameters (pre-norm, stochastic depth 0.1). ConvBlock = [Conv$3{\times}3$, BN, GELU]$\times$2.
$\oplus$ = channel concatenation (skip connection); $\uparrow$2 = bilinear upsampling. Clinical-token groups:
cup--disc (CDR, vCDR, hCDR), rim (I, S, N+T fractions, rim ratio, ISNT violation), size (disc area,
ellipticity), edge (disc and cup boundary strength). $r_j$ is the rim probability pooled to the $14{\times}14$
patch grid; $\alpha_l$ is learned per block.
\end{tablenotes}
\end{threeparttable}
\end{table}

%% ---------------------------------------------------------------------------------------- Table 5
\begin{table}[h]
\centering
\begin{threeparttable}
\caption{BT-ViT v1: training configuration.}
\label{tab:v1config}
\footnotesize
\begin{tabular}{@{}ll@{}}
\toprule
Setting & Value \\
\midrule
Training data & ORIGA + REFUGE: 928 labelled ROIs (159 glaucoma) + 800 segmentation-only ROIs \\
Model selection & Validation AUC (232 ROIs, 39 glaucoma); early stopping, patience 15 \\
Loss & $\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+\lambda_{\mathrm{seg}}\mathcal{L}_{\mathrm{seg}}+\lambda_{\mathrm{bio}}\mathcal{L}_{\mathrm{bio}}$;
CE; Dice + CE (3 classes); smooth-L1 to expert-mask biomarkers \\
Loss weights & $\lambda_{\mathrm{seg}}=1.0$, $\lambda_{\mathrm{bio}}=0.5$ \\
Optimiser & AdamW, weight decay 0.05 \\
Learning rate & $1{\times}10^{-4}$ (pretrained blocks), $3{\times}10^{-4}$ (new layers); one-cycle, 10\% warm-up \\
Epochs / batch & 50 (best epoch %%BESTEPOCH%%) / 16; fp32 \\
Sampling & Class-balanced; one third of each batch from segmentation-only images \\
Augmentation & Rotation $\pm20^{\circ}$, scale 0.9--1.1, centre jitter $\pm10\%$, flips; colour, gamma, blur, JPEG \\
Thresholds & Youden and maximum accuracy, both fitted on validation \\
Hardware & NVIDIA GTX 1650 Ti (4 GB); ${\approx}$27 s per epoch \\
\bottomrule
\end{tabular}
\end{threeparttable}
\end{table}

%% ---------------------------------------------------------------------------------------- Table 6
\begin{table}[h]
\centering
\begin{threeparttable}
\caption{BT-ViT v1: classification results (horizontal-flip test-time augmentation).}
\label{tab:v1results}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrc cccc cccc cc@{}}
\toprule
 & & & & \multicolumn{4}{c}{Youden threshold (%%THRY%%)} & \multicolumn{4}{c}{Max-accuracy threshold (%%THRA%%)} & & \\
\cmidrule(lr){5-8} \cmidrule(lr){9-12}
Set & $n$ & Pos. & AUC & Acc. & Sens. & Spec. & F1 & Acc. & Sens. & Spec. & F1 & ECE & TP/FN/FP/TN \\
\midrule
%%V1RESULTS%%
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item[] Both thresholds were fitted on the validation set and applied unchanged to the test and external
sets. The confusion counts (last column) refer to the Youden threshold. ECE, expected calibration error
(10 bins). Without test-time augmentation the internal-test AUC is 0.936 and RIM-ONE/DRISHTI-GS AUC
0.892/0.825.
\end{tablenotes}
\end{threeparttable}
\end{table}

%% ---------------------------------------------------------------------------------------- Table 7
\begin{table}[h]
\centering
\begin{threeparttable}
\caption{BT-ViT v1: segmentation produced by the model's own decoder.}
\label{tab:v1seg}
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{2}{c}{Optic disc} & \multicolumn{2}{c}{Optic cup} & \multicolumn{2}{c}{Biomarker error} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(l){6-7}
Set & Dice & IoU & Dice & IoU & vCDR & CDR \\
\midrule
%%V1SEG%%
\bottomrule
\end{tabular}
\begin{tablenotes}\footnotesize
\item[] Foreground-only Dice and IoU averaged per image. Biomarker error is the mean absolute difference
between the model's biomarkers and those computed from the expert masks.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{thebibliography}{2}\footnotesize
\bibitem{nasrin2026} A.~Nasrin \emph{et al.}, ``A lightweight vision transformer and retinal biomarker fusion
framework for early glaucoma detection,'' \emph{J. Clin. Med.}, vol.~15, p.~5651, 2026.
\bibitem{touvron2021} H.~Touvron, M.~Cord, M.~Douze, F.~Massa, A.~Sablayrolles and H.~J\'egou,
``Training data-efficient image transformers \& distillation through attention,'' in \emph{Proc. ICML},
2021, pp.~10347--10357.
\end{thebibliography}

\end{document}
"""


def main():
    import pandas as pd
    tex = TEMPLATE.replace("%%RESULTS%%", results_rows()).replace("%%EXPERIMENTS%%", experiments_rows())
    v1_rows, thr_y, thr_a = v1_results_rows("p2/BTViT_full/seed2026/metrics_tta.json")
    log = pd.read_csv(RUNS_DIR / "p2" / "BTViT_full" / "seed2026" / "train_log.csv")
    tex = (tex.replace("%%V1LAYERS%%", v1_layer_rows()).replace("%%V1RESULTS%%", v1_rows)
              .replace("%%THRY%%", f"{thr_y:.3f}").replace("%%THRA%%", f"{thr_a:.3f}")
              .replace("%%V1SEG%%", v1_seg_rows()).replace("%%BESTEPOCH%%", str(int(log.val_auc.idxmax()) + 1)))
    OUT.write_text(tex, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
