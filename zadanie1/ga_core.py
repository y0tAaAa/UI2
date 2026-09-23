from dataclasses import dataclass, field
from typing import Callable, Optional
import importlib

# Import dynamically so editors do not report a missing direct import in this
# module when NumPy is provided by the selected Python environment.
np = importlib.import_module("numpy")

import genetic_all as ga


@dataclass
class GAConfig:
    name: str                      # kratky identifikator experimentu, napr. "a"
    fitness_fn: Callable           # napr. ga.schwefel alebo ga.eggholder
    n_genes: int = 10
    lower: float = -500.0
    upper: float = 500.0
    pop_size: int = 50
    generations: int = 250

    # --- selekcia (selektivny tlak) ---
    # 'tournament' -> ga.seltourn (velky tlak)
    # 'roulette'   -> ga.selsus   (stredny tlak)
    # 'random'     -> ga.selrand  (ziadny/velmi maly tlak)
    selection: str = "tournament"
    elite_count: int = 1           # kolko najlepsich jedincov sa vzdy prenesie bez zmeny

    # --- krizenie ---
    crossover_on: bool = True
    cross_pts: int = 1
    cross_mode: int = 1            # 1 = nahodne dvojice, 0 = susedne dvojice

    # --- mutacie (diverzita) ---
    mutx_on: bool = True           # globalna mutacia (velky skok, novy nahodny gen)
    mutx_rate: float = 0.05
    muta_on: bool = True           # lokalna mutacia (maly aditivny posun)
    muta_rate: float = 0.08
    muta_amp: float = 20.0         # amplituda lokalnej mutacie (v jednotkach genu)

    seed: Optional[int] = None


def _select_parents(cfg: GAConfig, pop, fit, n):
    if cfg.selection == "tournament":
        new_pop, new_fit = ga.seltourn(pop, fit, n)
    elif cfg.selection == "roulette":
        new_pop, new_fit = ga.selsus(pop, fit, n)
    elif cfg.selection == "random":
        new_pop, new_fit = ga.selrand(pop, fit, n)
    else:
        raise ValueError(f"Neznamy typ selekcie: {cfg.selection}")
    return new_pop, new_fit


def run_ga(cfg: GAConfig):
    """
    Spusti jeden beh GA podla cfg.
    Vrati dict s historiou (best/mean fitness po generaciach) a najlepsim
    najdenym jedincom.
    """
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        import random
        random.seed(cfg.seed)

    space = ga.uniform_space(cfg.n_genes, cfg.lower, cfg.upper)
    amp = np.array([cfg.muta_amp] * cfg.n_genes)

    pop = ga.genrpop(cfg.pop_size, space)
    fit = cfg.fitness_fn(pop)

    best_hist = np.zeros(cfg.generations)
    mean_hist = np.zeros(cfg.generations)

    best_ever_x = None
    best_ever_f = np.inf

    for g in range(cfg.generations):
        fit = cfg.fitness_fn(pop)

        gen_best_idx = int(np.argmin(fit))
        if fit[gen_best_idx] < best_ever_f:
            best_ever_f = fit[gen_best_idx]
            best_ever_x = pop[gen_best_idx].copy()

        best_hist[g] = best_ever_f
        mean_hist[g] = float(np.mean(fit))

        # elitizmus: najlepsi jedinci sa nezmeneni prenesu do dalsej generacie
        if cfg.elite_count > 0:
            elite_pop, elite_fit = ga.selsort(pop, fit, cfg.elite_count)
        else:
            elite_pop = np.zeros((0, cfg.n_genes))

        n_offspring = cfg.pop_size - cfg.elite_count

        parents, _ = _select_parents(cfg, pop, fit, n_offspring)

        if cfg.crossover_on and n_offspring >= 2:
            parents = ga.crossov(parents, cfg.cross_pts, cfg.cross_mode)

        if cfg.mutx_on:
            parents = ga.mutx(parents, cfg.mutx_rate, space)

        if cfg.muta_on:
            parents = ga.muta(parents, cfg.muta_rate, amp, space)

        pop = np.vstack([elite_pop, parents]) if cfg.elite_count > 0 else parents

    return {
        "name": cfg.name,
        "best_hist": best_hist,
        "mean_hist": mean_hist,
        "best_x": best_ever_x,
        "best_f": best_ever_f,
    }


def run_multi(cfg: GAConfig, n_runs: int, base_seed: int = 0):
    """Spusti cfg n_runs-krat (s roznym seedom) a vrat pole best_hist (n_runs, generations)."""
    runs = []
    for i in range(n_runs):
        run_cfg = GAConfig(**{**cfg.__dict__, "seed": base_seed + i})
        result = run_ga(run_cfg)
        runs.append(result)
    best_matrix = np.stack([r["best_hist"] for r in runs])
    mean_matrix = np.stack([r["mean_hist"] for r in runs])
    return {
        "name": cfg.name,
        "runs": runs,
        "best_matrix": best_matrix,
        "mean_matrix": mean_matrix,
        "best_avg": best_matrix.mean(axis=0),
        "final_best": best_matrix[:, -1],
    }
