import numpy as np
import matplotlib.pyplot as plt

def MonthTimeSeries(Time, Vals, Title, YLabel, Legend,
                     Show=False, Window=25, PrintRunningIOA=False):

    n_vals = len(Vals)
    assert 1 <= n_vals <= 5, "Vals must contain between 1 and 5 elements."
    assert Window % 2 == 1, "Window must be odd."

    half_w = Window // 2

    ref = np.array(Vals[0])
    mean_diffs = []
    rmsds = []
    ioas = []

    # ===============================
    # BULK STATISTICS (UNCHANGED)
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
        rmsds.append(np.sqrt(np.mean(diff ** 2)))

        numerator = np.sum((model_valid - ref_valid) ** 2)
        denominator = np.sum(
            (np.abs(model_valid - np.mean(ref_valid)) +
             np.abs(ref_valid - np.mean(ref_valid))) ** 2
        )

        ioa = 1 - numerator / denominator if denominator != 0 else np.nan
        ioas.append(ioa)

    # ===============================
    # RUNNING IOA (UNCHANGED)
    # ===============================
    running_ioa = []
    model = np.array(Vals[1])

    for k in range(len(ref)):
        if k < half_w or k >= len(ref) - half_w:
            running_ioa.append(np.nan)
            continue

        ref_window = ref[k-half_w : k+half_w+1]
        model_window = model[k-half_w : k+half_w+1]

        valid_mask = (ref_window != -9999) & (~np.isnan(ref_window))

        ref_valid = ref_window[valid_mask]
        model_valid = model_window[valid_mask]

        if len(ref_valid) == 0:
            running_ioa.append(np.nan)
            continue

        numerator = np.sum((model_valid - ref_valid) ** 2)
        denominator = np.sum(
            (np.abs(model_valid - np.mean(ref_valid)) +
             np.abs(ref_valid - np.mean(ref_valid))) ** 2
        )

        ioa = 1 - numerator / denominator if denominator != 0 else np.nan
        running_ioa.append(ioa)

    running_ioa = np.array(running_ioa)

    # ===============================
    # DEBUG: PRINT (time, IOA)
    # ===============================
    if PrintRunningIOA:
        print("\n=== Running IOA (time, value) ===")
        for t, val in zip(Time, running_ioa):
            print(t, val)

    # ===============================
    # PLOTTING
    # ===============================
    if Show:
        fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

        colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']

        # === TOP: Time Series ===
        ax = axes[0]

        for i in range(1, n_vals):
            ax.plot(Time, Vals[i],
                    linewidth=1.5,
                    label=Legend[i],
                    color=colors[i-1])

        ref_mask = (ref != -9999) & (~np.isnan(ref))
        t_ref = np.array(Time)[ref_mask]
        v_ref = ref[ref_mask]

        ax.plot(t_ref, v_ref, '.k', label=Legend[0])

        ax.set_title(Title)
        ax.set_ylabel(YLabel)
        ax.grid(True)
        ax.legend()

        # === BOTTOM: Running IOA ===
        ax2 = axes[1]

        ax2.plot(Time, running_ioa, color='purple', linewidth=1.5)
        ax2.set_ylabel(f"{Window}-hr IOA")
        ax2.set_xlabel("Time")
        ax2.set_ylim(0, 1)
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    # ===============================
    # REQUIRED OUTPUT FORMAT
    # ===============================
    print([mean_diffs, rmsds, ioas])
    return [mean_diffs, rmsds, ioas]
