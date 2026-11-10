"""
Write the LaTeX tables of the article directly from the result CSVs, so no
number in the article is typed by hand.

  python src/report_tables.py      -> paper/tables/*.tex
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ORDER = ["face", "mc_bin", "mc_grey", "mc_smooth", "lookup", "rf", "pointnet", "cnn"]
NAME = {"face": "Voxel faces", "mc_bin": "MC (binary)", "mc_grey": "MC (greyscale)",
        "mc_smooth": "MC (smoothed)", "lookup": "MC + lookup", "rf": "Random Forest",
        "pointnet": "PointNet", "cnn": "3D CNN"}
GROUP_BREAK = {"lookup"}             # rule before the calibrated/learned block
OUT = os.path.join("paper", "tables")


def fmt(v, best, signed=False, nd=2):
    if pd.isna(v):
        return "--"
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    s = s.replace("-", "$-$").replace("+", "$+$")
    return f"\\textbf{{{s}}}" if best else s


def table(rows_by_method, cols, colfmt, header, caption, label, signed_cols=(), best_abs=True, place="t", wide=False):
    """wide=True: spans both columns of the two-column article (table*)."""
    env = "table*" if wide else "table"
    """rows_by_method: {method: [values per col]}"""
    methods = [m for m in ORDER if m in rows_by_method]
    vals = np.array([rows_by_method[m] for m in methods], dtype=float)
    lines = [f"\\begin{{{env}}}[{place}]", "  \\centering\\footnotesize", f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
             "  \\setlength{\\tabcolsep}{3.5pt}",
             f"  \\begin{{tabular}}{{{colfmt}}}", "    \\toprule", header, "    \\midrule"]
    for i, m in enumerate(methods):
        if m in GROUP_BREAK:
            lines.append("    \\midrule")
        cells = []
        for j in range(len(cols)):
            col = vals[:, j]
            key = np.abs(col) if (best_abs or j in signed_cols) else col
            best = np.nanmin(key) == key[i]
            cells.append(fmt(vals[i, j], best, signed=j in signed_cols))
        lines.append(f"    {NAME[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    ms = pd.read_csv("results/metrics_split.csv")

    # Main table: test MAPE / bias, held-out MAPE, both resolutions
    rows = {}
    for m in ms.method.unique():
        r = []
        for split, metric in (("test", "MAPE"), ("test", "bias"), ("held_super", "MAPE"), ("held_rough", "MAPE")):
            for N in (32, 64):
                v = ms[(ms.method == m) & (ms.split == split) & (ms.N == N)][metric]
                r.append(v.iloc[0] if len(v) else np.nan)
        rows[m] = r
    header = ("    & \\multicolumn{2}{c}{Test MAPE} & \\multicolumn{2}{c}{Test bias} "
              "& \\multicolumn{2}{c}{Superellipsoid} & \\multicolumn{2}{c}{Rough blob} \\\\\n"
              "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
              "    Method & $32^3$ & $64^3$ & $32^3$ & $64^3$ & $32^3$ & $64^3$ & $32^3$ & $64^3$ \\\\")
    with open(os.path.join(OUT, "main.tex"), "w") as f:
        f.write(table(rows, range(8), "lrrrrrrrr", header,
                      "Surface-area error (\\%) on held-out shapes, all noise levels pooled. MAPE: mean absolute "
                      "error; bias: mean signed error. The two right-hand blocks are shape families never seen "
                      "in training (MAPE). Best value per column in bold.",
                      "tab:main", signed_cols=(2, 3), wide=True))

    # Noise table (test MAPE)
    mn = pd.read_csv("results/metrics_noise.csv")
    rows = {}
    for m in mn.method.unique():
        rows[m] = [mn[(mn.method == m) & (mn.N == N) & np.isclose(mn.noise, s)].MAPE.iloc[0]
                   for N in (32, 64) for s in (0.0, 0.05, 0.15)]
    header = ("    & \\multicolumn{3}{c}{$32^3$} & \\multicolumn{3}{c}{$64^3$} \\\\\n"
              "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
              "    Method & $\\sigma_n{=}0$ & $0.05$ & $0.15$ & $0$ & $0.05$ & $0.15$ \\\\")
    with open(os.path.join(OUT, "noise.tex"), "w") as f:
        f.write(table(rows, range(6), "lrrrrrr", header,
                      "Test MAPE (\\%) by resolution and noise level. Best per column in bold.", "tab:noise"))

    # Diameter table (test MAPE), selected methods
    md = pd.read_csv("results/metrics_diam.csv")
    bins = ["<16", "16-24", "24-32", "32-48", "48-64"]
    rows = {}
    for m in md.method.unique():
        if m in ("face", "mc_bin"):
            continue
        rows[m] = [md[(md.method == m) & (md.diam_bin == b)].MAPE.iloc[0] for b in bins]
    header = "    Method & $<16$ & 16--24 & 24--32 & 32--48 & 48--64 \\\\"
    with open(os.path.join(OUT, "diameter.tex"), "w") as f:
        f.write(table(rows, range(5), "lrrrrr", header,
                      "Test MAPE (\\%) by object diameter in voxels (both resolutions pooled).", "tab:diameter"))

    # Family table: methods as columns would be wide; families as rows, selected methods
    mf = pd.read_csv("results/metrics_family.csv")
    sel = [m for m in ("mc_grey", "lookup", "rf", "pointnet", "cnn") if m in set(mf.method)]
    fam_order = ["sphere", "ellipsoid", "cylinder", "torus", "blob", "superellipsoid", "rough_blob"]
    fam_name = {"sphere": "Sphere", "ellipsoid": "Ellipsoid", "cylinder": "Cylinder", "torus": "Torus",
                "blob": "Smooth blob", "superellipsoid": "Superellipsoid$^\\dagger$", "rough_blob": "Rough blob$^\\dagger$"}
    lines = ["\\begin{table*}[t]", "  \\centering\\small",
             "  \\caption{MAPE (\\%) by shape family (test split and held-out families, all noise levels). "
             "$^\\dagger$Held out from training. Best per row and resolution in bold.}",
             "  \\label{tab:family}", "  \\setlength{\\tabcolsep}{3.5pt}\\footnotesize",
             "  \\resizebox{\\linewidth}{!}{\\begin{tabular}{l" + "r" * (2 * len(sel)) + "}", "    \\toprule",
             "    & " + " & ".join(f"\\multicolumn{{{len(sel)}}}{{c}}{{${N}^3$}}" for N in (32, 64)) + " \\\\",
             f"    \\cmidrule(lr){{2-{1 + len(sel)}}}\\cmidrule(lr){{{2 + len(sel)}-{1 + 2 * len(sel)}}}",
             "    Family & " + " & ".join([NAME[m].replace("MC (greyscale)", "MC grey")
                                         .replace("Random Forest", "RF")
                                         .replace("MC + lookup", "Lookup") for m in sel] * 2) + " \\\\",
             "    \\midrule"]
    for fam in fam_order:
        if fam == "superellipsoid":
            lines.append("    \\midrule")
        cells = []
        for N in (32, 64):
            v = [mf[(mf.family == fam) & (mf.N == N) & (mf.method == m)].MAPE.iloc[0] for m in sel]
            b = np.nanmin(v)
            cells += [fmt(x, x == b) for x in v]
        lines.append(f"    {fam_name[fam]} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table*}"]
    with open(os.path.join(OUT, "family.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # RF importance (inline list)
    imp = pd.read_csv("results/rf_feature_importance.csv", index_col=0).iloc[:, 0]
    with open(os.path.join(OUT, "rf_importance.tex"), "w") as f:
        nice = {"euler": "the Euler number", "noise_hat": "the noise estimate",
                "log_mcsmooth_over_mcgrey": "the smoothed-to-raw marching-cubes area ratio",
                "log_face_over_mcgrey": "the face-count-to-marching-cubes area ratio",
                "log_mcbin_over_mcgrey": "the binary-to-greyscale marching-cubes area ratio",
                "log_mcgrey_over_N2": "the marching-cubes area", "N": "the grid size",
                "log_compactness": "the compactness", "aspect1": "the first aspect ratio",
                "aspect2": "the second aspect ratio"}
        items = [f"{nice[k]} ({v:.2f})" for k, v in imp.head(4).items()]
        f.write(", ".join(items[:-1]) + " and " + items[-1] + ".\n")
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
