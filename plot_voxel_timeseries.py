import matplotlib.pyplot as plt
import streamlit as st

def plot_voxel_timeseries(data, coords=(32, 32, 15)):
    """Plots the timeseries of a voxel at given (x, y, z) coordinates."""
    if data.ndim != 4:
        st.error("This dataset is not 4D (time series). Cannot plot voxel.")
        return

    x, y, z = coords
    timeseries = data[x, y, z, :]

    fig, ax = plt.subplots()
    ax.plot(timeseries)
    ax.set_title(f"Voxel Time Series at {coords}")
    ax.set_xlabel("Time Points")
    ax.set_ylabel("Signal Intensity")
    st.pyplot(fig)
