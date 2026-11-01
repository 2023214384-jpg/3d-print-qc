import streamlit as st
import os
import pandas as pd
import datetime
from ultralytics import YOLO
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="3D Print Quality Control Dashboard", layout="wide", initial_sidebar_state="expanded")

# Robust cached model loader
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

def get_model():
    possible_paths = [
        "best.pt",
        os.path.join(os.path.dirname(__file__), "best.pt") if "__file__" in globals() else "best.pt",
        "/mount/src/3d-print-qc/best.pt"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return load_yolo_model(path)
    return None

model = get_model()

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Controls
st.sidebar.title("⚙️ Control Panel")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=30, max_value=90, value=50, step=5) / 100.0

if st.sidebar.button("🗑️ Reset All Log Data"):
    st.session_state.history = []
    st.rerun()

# Main Header
st.title("🏭 3D Printing Quality Assurance & Analytics Dashboard")
st.markdown("Automated defect detection system and real-time production monitoring.")

if model is None:
    st.error("❌ Model weight 'best.pt' not found in workspace. Please ensure 'best.pt' is uploaded to your main GitHub repository.")
else:
    # Compute KPI counters
    history_list = st.session_state.history
    total_scanned = len(history_list)
    no_defect_cnt = sum(1 for item in history_list if item["Status"] == "No Defect")
    crack_cnt = sum(1 for item in history_list if item["Status"] == "Crack")
    stringing_cnt = sum(1 for item in history_list if item["Status"] == "Stringing")
    uncertain_cnt = sum(1 for item in history_list if item["Status"] == "Uncertain")
    
    pass_rate = (no_defect_cnt / total_scanned * 100) if total_scanned > 0 else 0.0

    # Top Row: Metric Summary Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("📦 Total Scanned", total_scanned)
    kpi2.metric("✅ No Defect", no_defect_cnt)
    kpi3.metric("⚠️ Crack", crack_cnt)
    kpi4.metric("⚠️ Stringing", stringing_cnt)
    kpi5.metric("📈 Pass Rate", f"{pass_rate:.1f}%")

    st.divider()

    # Main Work Area
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📷 Inspection Unit")
        uploaded_file = st.file_uploader("Upload 3D Printed Part Image", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Sample", use_container_width=True)
            
            if st.button("🔍 Run Quality Inspection", type="primary"):
                results = model.predict(source=image, verbose=False)
                top1_id = results[0].probs.top1
                raw_label = results[0].names[top1_id]
                conf = float(results[0].probs.top1conf.item())
                
                label_lower = raw_label.lower().replace("_", " ")
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Apply Confidence Safeguard
                if conf < confidence_threshold:
                    status = "Uncertain"
                    st.warning(f"⚠️ **UNCERTAIN CLASSIFICATION** ({conf*100:.1f}% confidence < {confidence_threshold*100:.0f}% threshold). Flagged for review.")
                elif "crack" in label_lower:
                    status = "Crack"
                    st.error(f"🚨 **DEFECT DETECTED: CRACK** ({conf*100:.1f}% confidence)")
                elif "string" in label_lower:
                    status = "Stringing"
                    st.warning(f"⚠️ **DEFECT DETECTED: STRINGING** ({conf*100:.1f}% confidence)")
                else:
                    status = "No Defect"
                    st.success(f"✅ **PASSED: NO DEFECT** ({conf*100:.1f}% confidence)")

                # Append to history
                st.session_state.history.append({
                    "Timestamp": timestamp,
                    "Filename": uploaded_file.name,
                    "Status": status,
                    "Confidence": f"{conf*100:.1f}%",
                    "Raw Class": raw_label
                })
                st.rerun()

    with right_col:
        st.subheader("📊 Analytics Breakdown")
        if total_scanned > 0:
            chart_data = pd.DataFrame({
                "Category": ["No Defect", "Crack", "Stringing", "Uncertain"],
                "Count": [no_defect_cnt, crack_cnt, stringing_cnt, uncertain_cnt]
            })
            chart_data = chart_data[chart_data["Count"] > 0]
            
            fig = px.pie(
                chart_data, 
                values="Count", 
                names="Category", 
                title="Defect Distribution Tally",
                hole=0.4,
                color="Category",
                color_discrete_map={
                    "No Defect": "#2ecc71", 
                    "Crack": "#e74c3c", 
                    "Stringing": "#f39c12", 
                    "Uncertain": "#95a5a6"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inspection data logged yet. Upload an image to start counting.")

    st.divider()

    # Inspection Log Table
    st.subheader("📋 Inspection Log Table")
    if total_scanned > 0:
        df_log = pd.DataFrame(history_list)
        st.dataframe(df_log, use_container_width=True)
        
        csv_data = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Inspection Log (CSV)",
            data=csv_data,
            file_name="3D_Print_Quality_Log.csv",
            mime="text/csv"
        )
    else:
        st.caption("Scan history will populate dynamically in this table.")
