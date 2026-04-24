import numpy as np
import matplotlib.pyplot as plt

def MonthTimeSeries(Time, Vals, Title, YLabel, Legend,
                    Show=False, Window=25):

    n_vals = len(Vals)
    assert 1 <= n_vals <= 5
    assert Window % 2 == 1

    half_w = Window // 2

    ref = np.array(Vals[0])

    mean_diffs = []
    rmsds = []
    ioas = []

    # ===============================
    # BULK STATISTICS
    # ===============================
    for i in range(1, n_vals):
        model = np.array(Vals[i])

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
        rmsds.append(np.sqrt(np.mean(diff**2)))

        numerator = np.sum((model_valid - ref_valid) ** 2)
        denominator = np.sum(
            (np.abs(model_valid - np.mean(ref_valid)) +
             np.abs(ref_valid - np.mean(ref_valid))) ** 2
        )

        ioas.append(1 - numerator / denominator if denominator != 0 else np.nan)

    # ===============================
    # RUNNING IOA (NOW FOR ALL MODELS)
    # ===============================
    running_ioas = []

    for i in range(1, n_vals):
        model = np.array(Vals[i])
        r_ioa = []

        for k in range(len(ref)):
            if k < half_w or k >= len(ref) - half_w:
                r_ioa.append(np.nan)
                continue

            ref_w = ref[k-half_w:k+half_w+1]
            mod_w = model[k-half_w:k+half_w+1]

            valid_mask = (ref_w != -9999) & (~np.isnan(ref_w))

            ref_v = ref_w[valid_mask]
            mod_v = mod_w[valid_mask]

            if len(ref_v) == 0:
                r_ioa.append(np.nan)
                continue

            numerator = np.sum((mod_v - ref_v) ** 2)
            denominator = np.sum(
                (np.abs(mod_v - np.mean(ref_v)) +
                 np.abs(ref_v - np.mean(ref_v))) ** 2
            )

            r_ioa.append(1 - numerator / denominator if denominator != 0 else np.nan)

        running_ioas.append(np.array(r_ioa))

    # ===============================
    # PLOTTING
    # ===============================
    if Show:
        fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

        colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']

        # --- Time series ---
        ax = axes[0]

        for i in range(1, n_vals):
            ax.plot(Time, Vals[i],
                    linewidth=1.5,
                    label=Legend[i],
                    color=colors[i-1])

        ax.plot(Time, ref, '.k', label=Legend[0])

        ax.set_title(Title)
        ax.set_ylabel(YLabel)
        ax.grid(True)
        ax.legend()

        # --- Running IOA ---
        ax2 = axes[1]

        for i, r_ioa in enumerate(running_ioas):
            ax2.plot(Time, r_ioa,
                     linewidth=1.2,
                     label=f"{Legend[i+1]} IOA",
                     color=colors[i])

        ax2.set_ylabel(f"{Window}-hr IOA")
        ax2.set_xlabel("Time")
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    print([mean_diffs, rmsds, ioas])
    return [mean_diffs, rmsds, ioas]
