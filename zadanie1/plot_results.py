import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

RUN_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
AVG_COLOR = "#111111"


def load_group(path, keys):
    data = np.load(path)
    out = {}
    for k in keys:
        out[k] = {
            "best_matrix": data[f"{k}_best_matrix"],
            "best_avg": data[f"{k}_best_avg"],
        }
    return out


def plot_group(group, titles, suptitle, out_path, fn_optimum=None, ncols=3, share_ylim=True):
    keys = list(group.keys())
    n = len(keys)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows), squeeze=False)

    if share_ylim:
        all_vals = np.concatenate([group[k]["best_matrix"].ravel() for k in keys])
        ymin, ymax = all_vals.min(), all_vals.max()
        if fn_optimum is not None:
            ymin = min(ymin, fn_optimum)
        pad = 0.04 * (ymax - ymin if ymax > ymin else 1.0)
        ylim = (ymin - pad, ymax + pad)
    else:
        ylim = None

    for idx, key in enumerate(keys):
        ax = axes[idx // ncols][idx % ncols]
        best_matrix = group[key]["best_matrix"]
        best_avg = group[key]["best_avg"]
        n_runs, gens = best_matrix.shape
        x = np.arange(gens)

        for r in range(n_runs):
            ax.plot(x, best_matrix[r], color=RUN_COLORS[r % len(RUN_COLORS)],
                     linewidth=0.9, alpha=0.65)
        ax.plot(x, best_avg, color=AVG_COLOR, linewidth=2.2, label="priemer")

        if fn_optimum is not None:
            ax.axhline(fn_optimum, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_title(titles.get(key, key), fontsize=10.5)
        ax.set_xlabel("generácia")
        ax.set_ylabel("best fitness")
        ax.grid(alpha=0.25)
        if ylim is not None:
            ax.set_ylim(ylim)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


TITLES_AE = {
    "a": "a) veľký tlak + veľká diverzita",
    "b": "b) veľký tlak + malá diverzita",
    "c": "c) malý tlak + veľká diverzita",
    "d": "d) malý tlak + malá diverzita",
    "e": "e) kompromis",
}

TITLES_FI = {
    "e": "e) kompromis (referencia)",
    "f": "f) e bez kríženia",
    "g": "g) e bez globálnej mutácie (mutx)",
    "h": "h) e bez lokálnej mutácie (muta)",
    "i": "i) e bez oboch mutácií",
}

SCHWEFEL10_OPT = -4189.829


def main():
    ae = load_group(os.path.join(OUT_DIR, "schwefel_a_e.npz"), ["a", "b", "c", "d", "e"])
    plot_group(ae, TITLES_AE,
               "Schwefel-10: parametrizácia GA - selektívny tlak × diverzita (a-e)",
               os.path.join(OUT_DIR, "fig1_schwefel_a_e.png"),
               fn_optimum=SCHWEFEL10_OPT)

    fi = load_group(os.path.join(OUT_DIR, "schwefel_f_i.npz"), ["e", "f", "g", "h", "i"])
    plot_group(fi, TITLES_FI,
               "Schwefel-10: vypínanie operátorov vo variante e (e-i)",
               os.path.join(OUT_DIR, "fig2_schwefel_e_i.png"),
               fn_optimum=SCHWEFEL10_OPT)


if __name__ == "__main__":
    main()
