import numpy as np
import matplotlib.pyplot as plt

def MonthTimeSeries(Time, Vals, Title, YLabel, Legend, Show=False):
    # Ensure Vals has at least one series
    n_vals = len(Vals)
    assert 1 <= n_vals <= 5, "Vals must contain between 1 and 5 elements."

    ref = np.array(Vals[0])
    mean_diffs = []
    rmsds = []
    ioas = []

    for i in range(1, n_vals):
        model = np.array(Vals[i])

        # Mask to exclude -9999 from Vals[0]
        valid_mask = (ref != -9999) & (~np.isnan(ref))

        ref_valid = ref[valid_mask]
        model_valid = model[valid_mask]

        if len(ref_valid) == 0:
            mean_diffs.append(np.nan)
            rmsds.append(np.nan)
            ioas.append(np.nan)
            continue

        diff = model_valid - ref_valid
        mean_diffs.append(np.mean(diff))
        rmsds.append(np.sqrt(np.mean(diff ** 2)))

        numerator = np.sum((model_valid - ref_valid) ** 2)
        denominator = np.sum(
            (np.abs(model_valid - np.mean(ref_valid)) +
             np.abs(ref_valid - np.mean(ref_valid))) ** 2
        )
        ioa = 1 - numerator / denominator if denominator != 0 else np.nan
        ioas.append(ioa)

    if Show:
        fig, ax = plt.subplots(1, 1, figsize=(18, 4))  # wide + short

        colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']

        # === Plot model(s) ===
        for i in range(1, n_vals):
            ax.plot(Time, Vals[i],
                    linewidth=1.5,
                    label=Legend[i],
                    color=colors[i-1])

        # === Plot reference (CIMIS) ===
        ref_mask = (ref != -9999) & (~np.isnan(ref))
        t_ref = np.array(Time)[ref_mask]
        v_ref = ref[ref_mask]

        ax.plot(t_ref, v_ref, '.k', label=Legend[0])

        ax.set_title(Title)
        ax.set_ylabel(YLabel)
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()

    return [mean_diffs, rmsds, ioas]
