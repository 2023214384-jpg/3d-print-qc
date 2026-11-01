import streamlit as st
import os
import pandas as pd
import datetime
from ultralytics import YOLO
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="3D Print Quality Control Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    if os.path.exists("/content/best.pt"):
        return YOLO("/content/best.pt")
    elif os.path.exists("/content/drive/MyDrive/best.pt"):
        return YOLO("/content/drive/MyDrive/best.pt")
    for root, dirs, files in os.walk("/content"):
        if "best.pt" in files:
            return YOLO(os.path.join(root, "best.pt"))
    return None

model = load_model()

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Controls
st.sidebar.title("⚙️ Quality Control Panel")
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", min_value=30, max_value=90, value=60, step=5) / 100.0

if st.sidebar.button("🗑️ Reset All Log Data"):
    st.session_state.history = []
    st.rerun()

# Main Title & Header
st.title("🏭 3D Printing Quality Assurance & Analytics Dashboard")
st.markdown("Real-time automated defect detection and production monitoring system.")

if model is None:
    st.error("❌ Model weight 'best.pt' not found. Please ensure it is present in Google Drive or workspace.")
else:
    # Top Row: Metric Summary Cards
    df_history = pd.DataFrame(st.session_state.history)
    
    total_scanned = len(df_history)
    no_defect_cnt = len(df_history[df_history["Status"] == "No Defect"]) if total_scanned > 0 else 0
    crack_cnt = len(df_history[df_history["Status"] == "Crack"]) if total_scanned > 0 else 0
    stringing_cnt = len(df_history[df_history["Status"] == "Stringing"]) if total_scanned > 0 else 0
    uncertain_cnt = len(df_history[df_history["Status"] == "Uncertain"]) if total_scanned > 0 else 0
    
    pass_rate = (no_defect_cnt / total_scanned * 100) if total_scanned > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("📦 Total Scanned", total_scanned)
    kpi2.metric("✅ No Defect", no_defect_cnt)
    kpi3.metric("⚠️ Crack", crack_cnt)
    kpi4.metric("⚠️ Stringing", stringing_cnt)
    kpi5.metric("📈 Pass Rate", f"{pass_rate:.1f}%")

    st.divider()

    # Main Work Area: Split Screen
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📷 Product Inspection Unit")
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
                    st.warning(f"⚠️ **UNCERTAIN CLASSIFICATION** ({conf*100:.1f}% confidence < {confidence_threshold*100:.0f}% threshold). Flagged for manual review.")
                elif "crack" in label_lower:
                    status = "Crack"
                    st.error(f"🚨 **DEFECT DETECTED: CRACK** ({conf*100:.1f}% confidence)")
                elif "string" in label_lower:
                    status = "Stringing"
                    st.warning(f"⚠️ **DEFECT DETECTED: STRINGING** ({conf*100:.1f}% confidence)")
                else:
                    status = "No Defect"
                    st.success(f"✅ **PASSED: NO DEFECT** ({conf*100:.1f}% confidence)")

                # Record in History Log
                st.session_state.history.append({
                    "Timestamp": timestamp,
                    "Filename": uploaded_file.name,
                    "Status": status,
                    "Confidence": f"{conf*100:.1f}%",
                    "Raw Class": raw_label
                })
                st.rerun()

    with right_col:
        st.subheader("📊 Analytics & Quality Breakdown")
        if total_scanned > 0:
            # Interactive Donut Chart
            chart_data = pd.DataFrame({
                "Category": ["No Defect", "Crack", "Stringing", "Uncertain"],
                "Count": [no_defect_cnt, crack_cnt, stringing_cnt, uncertain_cnt]
            })
            chart_data = chart_data[chart_data["Count"] > 0]
            
            fig = px.pie(
                chart_data, 
                values="Count", 
                names="Category", 
                title="Defect Distribution Ratio",
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
            st.info("No inspection data available yet. Upload and inspect an image to generate analytics.")

    st.divider()

    # Bottom Row: Detailed Log Table & Export
    st.subheader("📋 Inspection History Log")
    if total_scanned > 0:
        st.dataframe(df_history, use_container_width=True)
        
        # CSV Export Button for Power BI
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Logs for Power BI Analysis (CSV)",
            data=csv_data,
            file_name="3D_Print_Quality_Logs.csv",
            mime="text/csv"
        )
    else:
        st.caption("Log table will populate dynamically as items are inspected.")
