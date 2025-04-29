import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import nibabel as nib
from nilearn import plotting

def plot_mean_bold_signal(data, affine):

    #Calculate mean of data
    mean_fmri_data = np.mean(data, axis=3)

    #Convert into nifti image
    mean_fmri_img = nib.Nifti1Image(mean_fmri_data, affine)

    #Plot
    fig = plotting.plot_epi(mean_fmri_img)
    st.pyplot(fig)

    # # Create plot
    # display = plotting.plot_epi(mean_fmri_img, display_mode='ortho')

    # # Retrieve the current matplotlib figure
    # fig = plt.gcf()

    # # Show plot in Streamlit
    # st.pyplot(fig)

    # # Close display to avoid memory leak
    # display.close()