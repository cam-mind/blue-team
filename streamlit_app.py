import streamlit as st
import nibabel as nib
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import tempfile
import os
import zipfile
import time
import openai
from dotenv import load_dotenv

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

# ------------------ Config ------------------
st.set_page_config(page_title="Multimodal Imaging Viewer", layout="wide")
st.image("banner.png", use_container_width=True)
st.title("Multimodal Imaging Hackathon Toolkit")

# ------------------ Google Drive ------------------
st.sidebar.header("Download from Google Drive")
drive_url = st.sidebar.text_input("Enter Google Drive file URL or ID")
download_button = st.sidebar.button("Download and Extract")

EXTRACT_DIR = "/tmp/extracted"
progress_placeholder = st.empty()

def download_and_extract_from_gdrive(file_id_or_url):
    if os.path.exists("/tmp/bigdata.zip"):
        os.remove("/tmp/bigdata.zip")
    if os.path.exists(EXTRACT_DIR):
        for f in os.listdir(EXTRACT_DIR):
            try:
                os.remove(os.path.join(EXTRACT_DIR, f))
            except:
                pass
    try:
        import gdown
        if "drive.google.com" in file_id_or_url:
            if "/d/" in file_id_or_url:
                file_id = file_id_or_url.split("/d/")[1].split("/")[0]
            elif "id=" in file_id_or_url:
                file_id = file_id_or_url.split("id=")[1].split("&")[0]
            else:
                st.error("Could not parse Google Drive link.")
                return None
        else:
            file_id = file_id_or_url

        dest_path = "/tmp/bigdata.zip"
        progress_placeholder.info("Downloading...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest_path, quiet=False)

        os.makedirs(EXTRACT_DIR, exist_ok=True)
        progress_placeholder.info("Extracting...")
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

        progress_placeholder.success("Download and extraction complete.")
        time.sleep(1)
        progress_placeholder.empty()
        return EXTRACT_DIR
    except Exception as e:
        progress_placeholder.error(f"Download failed: {e}")
        return None

if download_button and drive_url:
    extracted_path = download_and_extract_from_gdrive(drive_url)
    if extracted_path:
        st.success(f"Extracted to: {extracted_path}")

# ------------------ Local /data scanner ------------------
DATA_DIR = "./data"

def get_local_data_files():
    supported_exts = (".nii.gz", ".json", ".tsv", ".edf")
    local_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith(supported_exts):
                local_files.append(os.path.join(root, f))
    return local_files

local_data_files = get_local_data_files()

# ------------------ Select .nii.gz and .json from repo ------------------
st.sidebar.header("Select BOLD Imaging & Metadata from /data")

nii_files = [f for f in local_data_files if f.endswith(".nii.gz")]
json_files = [f for f in local_data_files if f.endswith(".json")]

selected_nii = st.sidebar.selectbox("Select BOLD .nii.gz file", nii_files if nii_files else ["No NIfTI files found"])
selected_json = st.sidebar.selectbox("Select accompanying .json file", json_files if json_files else ["No JSON files found"])

if selected_nii and selected_json and selected_nii.endswith(".nii.gz") and selected_json.endswith(".json"):
    st.success("✅ Selected both .nii.gz and .json files from repo")

    # Load NIfTI
    try:
        img = nib.load(selected_nii)
        data = img.get_fdata()
        affine = img.affine
        st.subheader("🧠 BOLD NIfTI Information")
        st.write("Shape:", data.shape)
        st.write("Affine:")
        st.code(str(affine))
    except Exception as e:
        st.error(f"Failed to load NIfTI: {e}")
        data = None
        affine = None

    # Load JSON
    try:
        with open(selected_json, 'r', encoding='utf-8') as jf:
            json_data = json.load(jf)
        st.subheader("📄 Metadata from JSON")
        st.json(json_data)
    except Exception as e:
        st.error(f"Could not parse JSON: {e}")
        json_data = None


# ------------------ Chat Assistant Interface ------------------
# ------------------ Chat Assistant Interface ------------------
st.header("💬 Neuroimaging Chat Assistant")

import openai

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key is missing. Please check your .env file.")
else:
    openai.api_key = api_key  # Correct for installed SDK

    # Load prompt safely
    try:
        with open("master_prompt.txt", "r", encoding="utf-8") as file:
            system_prompt = file.read().strip()
    except FileNotFoundError:
        st.warning("No master_prompt.txt found. Using default prompt.")
        system_prompt = (
            "You are a helpful assistant for interpreting neuroimaging data. "
            "Use imaging shapes, metadata, and task context to assist the user."
        )

    def safe_openai_call(system_prompt, context, user_query):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{context}\n\n{user_query}"},
                ],
                max_tokens=512,
                temperature=0.7,
                request_timeout=30  # <-- Force timeout safeguard
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI API failed: {e}")
            return "Sorry, something went wrong with the assistant. Please try again later."

    # Chat input
    user_query = st.text_input("Ask something about your neuroimaging data:")

    if st.button("Send") and user_query.strip():
        # Build context safely
        context_lines = []
        if "data" in locals() and data is not None:
            context_lines.append(f"NIfTI shape: {data.shape}")
        if "affine" in locals() and affine is not None:
            context_lines.append(f"Affine: {affine.tolist()}")
        if "json_data" in locals() and json_data is not None:
            context_lines.append(f"Metadata: {json.dumps(json_data, indent=2)}")
        context = "\n\n".join(context_lines)

        # Safe call
        reply = safe_openai_call(system_prompt, context, user_query)

        st.text_area("Assistant Response:", value=reply, height=200)

#### NOW THE SECTION WHERE WE ACTUALLY IMPORT THE SCRIPTS WE WANT TO CALL
#### FOR NOW I HAVE DONE THE FOLLOWING : the user will have first a fixed set of options to select from 
#### and only later access to the chatbox to have more ad-hoc queries with the model.
from analysis_and_visualisation.plot_voxel_timeseries import plot_voxel_timeseries
from analysis_and_visualisation.plot_mean_bold_signal import plot_mean_bold_signal
from analysis_and_visualisation.plot_bold_signal_across_time import plot_bold_signal_across_time
# Import others as needed
st.header("🔍 Choose an Action on the Imaging Data")

actions = [
    "Plot Voxel Time Series",
    "Plot Mean BOLD Signal",
    "Extract Timecourse from ROI (Coming Soon)", #<- not yet implemented
    "Plot BOLD signal across time"
]

selected_action = st.selectbox("Select an analysis to perform:", actions)

if st.button("Run Selected Analysis"):
    if selected_action == "Plot Voxel Time Series":
        # Ask user for voxel coordinates
        x = st.number_input("X coordinate", min_value=0, value=32)
        y = st.number_input("Y coordinate", min_value=0, value=32)
        z = st.number_input("Z coordinate", min_value=0, value=15)
        plot_voxel_timeseries(data, coords=(x, y, z))

    elif selected_action == "Plot Mean BOLD Signal":
        
        plot_mean_bold_signal(data, affine)
    
    elif selected_action =='Plot BOLD Signal across time':
        st.title("fMRI Viewer")

        # Slider
        timepoint = st.slider(
            "Select Timepoint",
            min_value=0,
            max_value=data.shape[-1] - 1,
            value=0
        )
        # Call plotting function from external script and show figure
        fig = plot_bold_signal_across_time(data, timepoint, affine)
        st.pyplot(fig)



    else:
        st.info("This action is not yet implemented.")

