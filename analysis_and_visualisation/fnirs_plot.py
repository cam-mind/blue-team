import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import h5py

def plot_multiple_fnirs_channels(raw, duration=10.0, start=0.0, n_channels=8, variance_threshold=1e-6):
    """
    Plots several fNIRS channels and flags low-variance (potentially bad) ones.

    Parameters:
    - raw: MNE Raw object
    - duration: seconds of data to display
    - start: start time in seconds
    - n_channels: number of channels to plot
    - variance_threshold: below this, a channel is flagged as bad
    """

    st.subheader("📊 fNIRS Channel Quality Check")

    sfreq = raw.info['sfreq']
    start_sample = int(start * sfreq)
    end_sample = int((start + duration) * sfreq)

    data = raw.get_data()[:, start_sample:end_sample]
    times = raw.times[start_sample:end_sample]
    ch_names = raw.ch_names

    fig, ax = plt.subplots(figsize=(10, 6))

    bad_channels = []

    for idx in range(min(n_channels, data.shape[0])):
        signal = data[idx]
        var = np.var(signal)

        label = f"{ch_names[idx]} (Var={var:.2e})"

        if var < variance_threshold:
            bad_channels.append(ch_names[idx])
            ax.plot(times, signal + idx * 1e-6, color='red', label=label)  # offset & red for bad
        else:
            ax.plot(times, signal + idx * 1e-6, label=label)

    ax.set_title(f"First {n_channels} Channels (offset vertically, {duration}s from {start}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Offset Signal")
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)
    st.pyplot(fig)

    if bad_channels:
        st.warning(f"⚠️ Flagged low-variance channels: {', '.join(bad_channels)}")
    else:
        st.success("✅ All displayed channels passed variance check.")