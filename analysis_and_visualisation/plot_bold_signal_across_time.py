import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import nilearn
import nibabel as nib
from nilearn import plotting


def plot_bold_signal_across_time(fmri_data, timepoint, affine):
    volume = fmri_data[..., timepoint]
    volume_img = nib.Nifti1Image(volume, affine)

    display = plotting.plot_epi(
        volume_img,
        title=f"Timepoint {timepoint}",
        display_mode='ortho',
        cut_coords=(0, 0, 0),
        colorbar=True
    )
    fig = plt.gcf()
    display.close()
    return fig
