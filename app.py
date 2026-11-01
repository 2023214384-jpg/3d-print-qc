import streamlit as st
import os
import pandas as pd
import datetime
from ultralytics import YOLO
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="3D Print Quality Control Dashboard", layout="wide", initial_sidebar_state="expanded")

# Robust cached loader
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

# Dynamically locate best.pt on every rerun
def get_model():
    possible_paths = [
        "best.pt",
        os.path.join(os.path.dirname(__file__), "best.pt"),
        "/mount/src/3d-print-qc/best.pt"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return load_yolo_model(path)
    return None

model = get_model()
