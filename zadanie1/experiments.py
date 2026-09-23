import os
import numpy as np
import genetic_all as ga
from ga_core import GAConfig, run_multi

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

POP_SIZE = 50
GENERATIONS = 250
N_RUNS = 6
BASE_SEED = 42

HIGH_PRESSURE = dict(selection="tournament", elite_count=3)
LOW_PRESSURE  = dict(selection="random",     elite_count=0)

HIGH_DIVERSITY = dict(mutx_rate=0.15, muta_rate=0.25, muta_amp=60.0)
LOW_DIVERSITY  = dict(mutx_rate=0.01, muta_rate=0.02, muta_amp=5.0)

COMPROMISE_SELECTION = dict(selection="tournament", elite_count=1)
COMPROMISE_DIVERSITY = dict(mutx_rate=0.05, muta_rate=0.08, muta_amp=20.0)


def base_cfg(name, fitness_fn=ga.schwefel, **overrides):
    cfg = GAConfig(
        name=name,
        fitness_fn=fitness_fn,
        n_genes=10,
        lower=-500.0,
        upper=500.0,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_configs_a_to_e(fitness_fn=ga.schwefel):
    configs = {}
    configs["a"] = base_cfg("a: vysoky tlak, velka diverzita", fitness_fn,
                             **HIGH_PRESSURE, **HIGH_DIVERSITY)
    configs["b"] = base_cfg("b: vysoky tlak, mala diverzita", fitness_fn,
                             **HIGH_PRESSURE, **LOW_DIVERSITY)
    configs["c"] = base_cfg("c: nizky tlak, velka diverzita", fitness_fn,
                             **LOW_PRESSURE, **HIGH_DIVERSITY)
    configs["d"] = base_cfg("d: nizky tlak, mala diverzita", fitness_fn,
                             **LOW_PRESSURE, **LOW_DIVERSITY)
    configs["e"] = base_cfg("e: kompromis", fitness_fn,
                             **COMPROMISE_SELECTION, **COMPROMISE_DIVERSITY)
    return configs


def build_configs_f_to_i(fitness_fn=ga.schwefel):
    configs = {}
    configs["f"] = base_cfg("f: e bez krizenia", fitness_fn,
                             **COMPROMISE_SELECTION, **COMPROMISE_DIVERSITY,
                             crossover_on=False)
    configs["g"] = base_cfg("g: e bez globalnej mutacie (mutx)", fitness_fn,
                             **COMPROMISE_SELECTION, **COMPROMISE_DIVERSITY,
                             mutx_on=False)
    configs["h"] = base_cfg("h: e bez lokalnej mutacie (muta)", fitness_fn,
                             **COMPROMISE_SELECTION, **COMPROMISE_DIVERSITY,
                             muta_on=False)
    configs["i"] = base_cfg("i: e bez oboch mutacii", fitness_fn,
                             **COMPROMISE_SELECTION, **COMPROMISE_DIVERSITY,
                             mutx_on=False, muta_on=False)
    return configs


def run_group(configs, base_seed=BASE_SEED, n_runs=N_RUNS):
    results = {}
    for key, cfg in configs.items():
        print(f"  running {key}: {cfg.name} ...")
        results[key] = run_multi(cfg, n_runs=n_runs, base_seed=base_seed)
    return results


def save_results(results, path):
    payload = {}
    for key, r in results.items():
        payload[f"{key}_best_matrix"] = r["best_matrix"]
        payload[f"{key}_mean_matrix"] = r["mean_matrix"]
        payload[f"{key}_best_avg"] = r["best_avg"]
        payload[f"{key}_final_best"] = r["final_best"]
    np.savez(path, **payload)


if __name__ == "__main__":
    print("== Schwefel-10: experimenty a-e (parametrizacia tlak/diverzita) ==")
    configs_ae = build_configs_a_to_e(ga.schwefel)
    results_ae = run_group(configs_ae)
    save_results(results_ae, os.path.join(OUT_DIR, "schwefel_a_e.npz"))

    print("== Schwefel-10: experimenty f-i (ablacia operatorov, zaklad = e) ==")
    configs_fi = build_configs_f_to_i(ga.schwefel)
    configs_fi_with_e = {"e": configs_ae["e"], **configs_fi}
    results_fi = run_group(configs_fi_with_e)
    save_results(results_fi, os.path.join(OUT_DIR, "schwefel_f_i.npz"))

    print("Hotovo. Vysledky v:", OUT_DIR)
    for key in results_ae:
        print(f"  Schwefel {key}: priemerny final best = {results_ae[key]['final_best'].mean():.3f}")
    for key in results_fi:
        print(f"  Schwefel {key}: priemerny final best = {results_fi[key]['final_best'].mean():.3f}")
