
import os
import numpy as np
import genetic_all as ga

from experiments import (
    build_configs_a_to_e, run_group, save_results, OUT_DIR, BASE_SEED, N_RUNS,
)
from plot_results import load_group, plot_group, TITLES_AE

EGG_LOWER, EGG_UPPER = -512.0, 512.0


def build_configs_eggholder():
    configs = build_configs_a_to_e(ga.eggholder)
    for cfg in configs.values():
        cfg.lower = EGG_LOWER
        cfg.upper = EGG_UPPER
    return configs


if __name__ == "__main__":
    print("== Eggholder-10: experimenty a-e ==")
    configs = build_configs_eggholder()
    results = run_group(configs, base_seed=BASE_SEED, n_runs=N_RUNS)
    save_results(results, os.path.join(OUT_DIR, "eggholder_a_e.npz"))

    for key in results:
        print(f"  Eggholder {key}: priemerny final best = {results[key]['final_best'].mean():.3f}")

    egg = load_group(os.path.join(OUT_DIR, "eggholder_a_e.npz"), ["a", "b", "c", "d", "e"])
    plot_group(egg, TITLES_AE,
               "Eggholder-10: parametrizácia GA - selektívny tlak × diverzita (a-e)",
               os.path.join(OUT_DIR, "fig3_eggholder_a_e.png"),
               fn_optimum=None)

    schwef = np.load(os.path.join(OUT_DIR, "schwefel_a_e.npz"))
    print("\n== Porovnanie priemerneho final-best fitness (mensie = lepsie) ==")
    print(f"{'var':4s} {'Schwefel-10':>14s} {'Eggholder-10':>14s}")
    for key in "abcde":
        sw = schwef[f"{key}_final_best"].mean()
        eg = results[key]["final_best"].mean()
        print(f"{key:4s} {sw:14.2f} {eg:14.2f}")
