import streamlit as st
import mne

def plot_topomap_at_timepoint(raw, timepoint = 5):
    eeg_channel_names = [ch for ch in raw.ch_names if ch in [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz'
    ]]
    raw.pick_channels(eeg_channel_names)
    raw.set_montage('standard_1020')  # Try ‘biosemi64’ if that fits your cap
    # Get data at the timepoint
    data, times = raw[:, int(raw.info['sfreq'] * timepoint)]
    # Plot topomap for that timepoint
    fig = mne.viz.plot_topomap(data[:, 0], raw.info, show=True)
    st.pyplot(fig)