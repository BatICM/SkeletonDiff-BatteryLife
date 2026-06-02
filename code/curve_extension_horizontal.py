import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class SOHCurveExtension:
    """
    Extend SOH curves by holding the last observed SOH value.

    This keeps the supervision target conservative and lets `loss_mask`
    decide which positions participate in training.
    """

    def __init__(self, min_soh=None, max_extension_ratio=0.5, soh_bounds=None):
        if soh_bounds is None:
            lower_bound = 0.0 if min_soh is None else float(min_soh)
            soh_bounds = (lower_bound, 100.0)
        self.soh_bounds = tuple(float(v) for v in soh_bounds)
        self.max_extension_ratio = max_extension_ratio

    def extend_soh_curve(self, cycles, soh, target_length, fit_start_ratio=0.7, verbose=True):
        cycles = np.asarray(cycles, dtype=float).flatten()
        soh = np.asarray(soh, dtype=float).flatten()

        valid_mask = ~(np.isnan(cycles) | np.isnan(soh))
        cycles = cycles[valid_mask]
        soh = soh[valid_mask]

        if len(cycles) == 0:
            raise ValueError("No valid SOH points for extension.")

        sort_idx = np.argsort(cycles)
        cycles = cycles[sort_idx]
        soh = soh[sort_idx]
        cycles, unique_idx = np.unique(cycles, return_index=True)
        soh = soh[unique_idx]

        lower_soh, upper_soh = self.soh_bounds
        soh = np.clip(soh, lower_soh, upper_soh)

        original_length = len(cycles)
        original_max_cycle = int(cycles[-1])

        if original_max_cycle >= target_length:
            if verbose:
                print(f"  original_max_cycle={original_max_cycle} >= target_length={target_length}, truncate only")
            target_cycles = np.arange(1, target_length + 1)
            f_interp = interp1d(
                cycles,
                soh,
                kind="linear",
                bounds_error=False,
                fill_value=(soh[0], soh[-1]),
            )
            final_soh = np.clip(f_interp(target_cycles), lower_soh, upper_soh)
            return target_cycles, final_soh, {
                "extended": False,
                "truncated": True,
                "method": "truncate",
                "original_length": original_length,
                "original_max_cycle": original_max_cycle,
                "target_length": int(target_length),
            }

        extension_needed = int(target_length - original_max_cycle)
        extension_ratio = extension_needed / max(original_max_cycle, 1)
        hold_value = float(soh[-1])

        if verbose:
            print(
                f"  horizontal extension: {original_max_cycle} -> {target_length} "
                f"(hold SOH={hold_value:.3f}, ratio={extension_ratio:.1%})"
            )

        ext_cycles = np.arange(original_max_cycle + 1, target_length + 1)
        ext_soh = np.full(ext_cycles.shape, hold_value, dtype=float)

        extended_cycles = np.concatenate([cycles, ext_cycles])
        extended_soh = np.concatenate([soh, ext_soh])

        target_cycles = np.arange(1, target_length + 1)
        f_interp = interp1d(
            extended_cycles,
            extended_soh,
            kind="linear",
            bounds_error=False,
            fill_value=(extended_soh[0], extended_soh[-1]),
        )
        final_soh = np.clip(f_interp(target_cycles), lower_soh, upper_soh)

        return target_cycles, final_soh, {
            "extended": True,
            "truncated": False,
            "method": "horizontal_hold",
            "hold_value": hold_value,
            "original_length": original_length,
            "original_max_cycle": original_max_cycle,
            "extended_length": len(extended_cycles),
            "extension_ratio": extension_ratio,
            "target_length": int(target_length),
            "exceeds_max_extension_ratio": (
                self.max_extension_ratio is not None and extension_ratio > self.max_extension_ratio
            ),
        }

    def extend_feature_matrix(self, feature_matrix, original_cycles, target_length, method="last_value"):
        feature_matrix = np.asarray(feature_matrix, dtype=float)
        original_cycles = np.asarray(original_cycles, dtype=float).flatten()

        if feature_matrix.ndim != 2:
            raise ValueError("feature_matrix must be 2D")

        original_max_cycle = int(original_cycles[-1])
        if original_max_cycle >= target_length:
            return feature_matrix

        extension_length = int(target_length - original_max_cycle)
        if method == "mean":
            pad_values = feature_matrix.mean(axis=1, keepdims=True)
        else:
            pad_values = feature_matrix[:, -1:]

        extension = np.repeat(pad_values, extension_length, axis=1)
        return np.concatenate([feature_matrix, extension], axis=1)


def visualize_extension(battery_id, original_cycles, original_soh, extended_cycles, extended_soh, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(original_cycles, original_soh, "b-", linewidth=2, label="Original", marker="o", markersize=2)

    ext_mask = np.asarray(extended_cycles) > np.asarray(original_cycles)[-1]
    if np.any(ext_mask):
        ax.plot(
            np.asarray(extended_cycles)[ext_mask],
            np.asarray(extended_soh)[ext_mask],
            "r--",
            linewidth=2,
            label="Horizontal extension",
        )
        ax.axvline(x=np.asarray(original_cycles)[-1], color="gray", linestyle=":", linewidth=1.5, label="Extension start")

    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"Battery {battery_id} SOH extension")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(65, 105)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"extension_{battery_id}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_battery_with_extension(battery_data, target_length, n_feature_cycles=100, matrix_size=100, verbose=True):
    summary = battery_data.get("summary", {})
    if "SOH_processed" not in summary or "cycle" not in summary:
        return None

    soh = np.asarray(summary["SOH_processed"], dtype=float)
    cycles = np.asarray(summary["cycle"], dtype=float)
    cycle_life = int(battery_data["cycle_life"][0][0])

    extender = SOHCurveExtension(min_soh=None, max_extension_ratio=0.5, soh_bounds=(0.0, 100.0))
    extended_cycles, extended_soh, ext_info = extender.extend_soh_curve(cycles, soh, target_length, verbose=verbose)

    seq_length = target_length // 10
    cycle_grid = np.linspace(target_length / seq_length, target_length, seq_length)
    f_interp = interp1d(
        extended_cycles,
        extended_soh,
        kind="linear",
        bounds_error=False,
        fill_value=(extended_soh[0], extended_soh[-1]),
    )
    soh_interp = f_interp(cycle_grid)
    soh_norm = (soh_interp / 100.0) * 2.0 - 1.0

    return {
        "soh_tensor": soh_norm.astype(np.float32),
        "raw_soh": extended_soh,
        "raw_cycles": extended_cycles,
        "cycle_life": cycle_life,
        "extension_info": ext_info,
        "target_length": target_length,
        "seq_length": seq_length,
    }
