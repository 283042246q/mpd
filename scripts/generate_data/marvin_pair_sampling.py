"""NumPy-only placement marginals and balanced bimanual pair distributions."""
import numpy as np


def mixture_distribution(config, arm, names, volumes):
    cfg = config["placement_mixture"]
    mix = np.asarray([cfg[k] for k in ("effective_volume", "uniform", "hard")], dtype=float)
    rates = cfg["collision_valid_rates"][arm]
    hard = cfg["hard_regions"][arm]
    if set(rates) != set(names) or not set(hard) <= set(names) or len(set(hard)) != len(hard):
        raise ValueError("placement_mixture region names must match the configured cells")
    r = np.asarray([rates[n] for n in names], dtype=float)
    if not np.isfinite(mix).all() or np.any(mix < 0) or not np.isclose(mix.sum(), 1):
        raise ValueError("placement_mixture weights must be nonnegative and sum to 1")
    if not np.isfinite(r).all() or np.any((r < 0) | (r > 1)):
        raise ValueError("collision_valid_rates must lie in [0, 1]")
    effective = np.asarray(volumes) * r
    h = np.asarray([n in hard for n in names], dtype=float)
    if effective.sum() <= 0 or h.sum() <= 0:
        raise ValueError("placement_mixture requires positive effective volume and a hard pool")
    return mix[0] * effective / effective.sum() + mix[1] / len(names) + mix[2] * h / h.sum()


def balance(kernel, rows, cols):
    """Iterative proportional fitting; fail explicitly for infeasible marginals."""
    matrix = np.asarray(kernel, dtype=float).copy()
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("pair preferences must be finite and nonnegative")
    for _ in range(10000):
        rs = matrix.sum(axis=1)
        if np.any(rs <= 0):
            raise ValueError("pair matrix has an unsupported row")
        matrix *= (rows / rs)[:, None]
        cs = matrix.sum(axis=0)
        if np.any(cs <= 0):
            raise ValueError("pair matrix has an unsupported column")
        matrix *= (cols / cs)[None, :]
        if max(np.max(np.abs(matrix.sum(1) - rows)), np.max(np.abs(matrix.sum(0) - cols))) < 1e-12:
            return matrix
    raise ValueError("pair matrix cannot satisfy the requested marginals")


def pair_distributions(config, distributions):
    cfg = config.get("placement_pair_sampling")
    if cfg is None:
        return None
    if config.get("placement_goal_must_differ_from_source_region", True):
        raise ValueError("placement_pair_sampling requires same-region motions enabled")
    diagonal = float(cfg["same_cell_fraction"])
    if not 0 < diagonal < 1:
        raise ValueError("same_cell_fraction must lie strictly between 0 and 1")
    prefs = cfg["preferences"]
    if any(not np.isfinite(v) or v <= 0 for v in prefs.values()):
        raise ValueError("pair preferences must be finite and positive")
    matrices = {}
    flags = {}
    for arm, (names, p) in distributions.items():
        if max(p) >= .5:
            raise ValueError("pair matrix needs max endpoint probability < 0.5; recalibrate mixture or diagonal policy")
        roles = [n[len(arm)+1:] for n in names]
        allowed = {"table", "table_cross_y", "table_cross_x", "cabinet", "cabinet_lower_xedge", "cabinet_lower_yedge", "cabinet_upper"}
        if set(roles) != allowed:
            raise ValueError("placement_pair_sampling requires the seven standard Marvin cells")
        cross = np.array([r.startswith("table_cross") for r in roles])
        upper = np.array([r == "cabinet_upper" for r in roles])
        kernel = np.zeros((len(p), len(p)))
        for i, a in enumerate(roles):
            for j, b in enumerate(roles):
                if i == j:
                    continue
                if "table" in (a, b):
                    key = "table_other"
                elif cross[i] and cross[j]:
                    key = "cross_cross"
                elif cross[i] or cross[j]:
                    key = "cross_shelf"
                elif upper[i] or upper[j]:
                    key = "lower_upper"
                else:
                    key = "lower_lower"
                kernel[i, j] = prefs[key]
        matrices[arm] = balance(kernel, (1-diagonal)*p, (1-diagonal)*p) + np.diag(diagonal*p)
        flags[arm] = ((cross[:, None] | cross[None, :]).ravel(), (upper[:, None] | upper[None, :]).ravel())
    coupling = cfg["dual_preferences"]
    if any(not np.isfinite(v) or not 0 < v <= 1 for v in coupling.values()):
        raise ValueError("dual preferences must lie in (0, 1]")
    lc, lu = flags["left"]
    rc, ru = flags["right"]
    kernel = np.ones((lc.size, rc.size))
    for mask, key in ((lc[:, None] & rc[None, :], "both_cross"),
                      (lu[:, None] & ru[None, :], "both_upper"),
                      ((lc[:, None] & ru[None, :]) | (lu[:, None] & rc[None, :]), "cross_upper")):
        kernel[mask] = np.minimum(kernel[mask], coupling[key])
    matrices["dual"] = balance(kernel, matrices["left"].ravel(), matrices["right"].ravel())
    return matrices


def pair_report(config, distributions):
    matrices = pair_distributions(config, distributions)
    if matrices is None:
        return None
    return {
        "semantics": "task proposal probabilities; finite accepted datasets may differ",
        "arms": {
            arm: {"region_order": list(names), "endpoint_probabilities": p.tolist(),
                  "pair_probabilities": matrices[arm].tolist(),
                  "same_cell_probability": float(np.trace(matrices[arm]))}
            for arm, (names, p) in distributions.items()
        },
        "dual_preferences": config["placement_pair_sampling"]["dual_preferences"],
        "dual_marginal_max_error": float(max(
            np.max(abs(matrices["dual"].sum(1) - matrices["left"].ravel())),
            np.max(abs(matrices["dual"].sum(0) - matrices["right"].ravel())))),
    }
