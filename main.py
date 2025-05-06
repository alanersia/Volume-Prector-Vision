import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Foreground Detection", layout="wide")

# ==================== Helpers ====================
def compute_occupancy(diff_mask):
    filled = cv2.countNonZero(diff_mask)
    total = diff_mask.shape[0] * diff_mask.shape[1]
    return (filled / total) * 100 if total > 0 else 0

# ==================== Session State Initialization ====================
for key, default in {
    'ref_bg': None,
    'mode': None,
    'cap': None,
    'history': pd.DataFrame({
        'timestamp': pd.Series(dtype='datetime64[ns]'),
        'occupancy': pd.Series(dtype='float')
    }),
    'last_log_time': time.time()
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==================== Sidebar Configuration ====================
st.sidebar.title("Konfigurasi")
st.session_state.mode = st.sidebar.selectbox(
    "Pilih Mode", ["Upload Video", "Webcam"], key="mode_select"
)
reset_btn = st.sidebar.button("Reset Reference Background", key="reset_ref")
update_btn = st.sidebar.button("Update Reference", key="update_ref")
stop_btn = st.sidebar.button("Stop Stream", key="stop_btn")
# Filter history display
time_filter = st.sidebar.selectbox(
    "Tampilkan data per", ['Per Hari', 'Per Bulan', 'Per Tahun'], key='filter_select'
)

if reset_btn:
    st.session_state.ref_bg = None
    st.session_state.history = pd.DataFrame({
        'timestamp': pd.Series(dtype='datetime64[ns]'),
        'occupancy': pd.Series(dtype='float')
    })
    st.sidebar.success("Reference background dan history di-reset.")

# ==================== Video Source Initialization ====================
if stop_btn and st.session_state.cap:
    st.session_state.cap.release()
    st.session_state.cap = None

if st.session_state.mode == "Upload Video":
    uploaded = st.sidebar.file_uploader("Upload video", type=["mp4","avi","mov"], key="uploader")
    if uploaded and not st.session_state.cap:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp.write(uploaded.read())
        st.session_state.cap = cv2.VideoCapture(tmp.name, cv2.CAP_FFMPEG)
elif st.session_state.mode == "Webcam":
    if not st.session_state.cap:
        st.session_state.cap = cv2.VideoCapture(0)

cap = st.session_state.cap

# ==================== Layout ====================
col1, col2 = st.columns(2)
with col1:
    st.header("Original View")
    orig_placeholder = st.empty()
with col2:
    st.header("Foreground Mask")
    mask_placeholder = st.empty()

chart_placeholder = st.empty()

# ==================== Continuous Loop ====================
if cap and cap.isOpened():
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    threshold = 30
    while True:
        if stop_btn:
            break
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if st.session_state.ref_bg is None:
            st.session_state.ref_bg = gray.copy()
        if update_btn:
            st.session_state.ref_bg = gray.copy()
        diff = cv2.absdiff(st.session_state.ref_bg, gray)
        _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        orig_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        mask_placeholder.image(cv2.cvtColor(fg, cv2.COLOR_GRAY2RGB), use_container_width=True)
        now = time.time()
        if now - st.session_state.last_log_time >= 1:
            occ = compute_occupancy(fg)
            ts = pd.to_datetime([datetime.now()])
            st.session_state.history = pd.concat([
                st.session_state.history,
                pd.DataFrame({'timestamp': ts, 'occupancy': [occ]})
            ], ignore_index=True)
            st.session_state.last_log_time = now
        # Prepare filtered df for chart
        df = st.session_state.history.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        current = datetime.now()
        if time_filter == 'Per Hari':
            start = current.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_filter == 'Per Bulan':
            start = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            start = current.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        df = df[df['timestamp'] >= start]
        df = df.set_index('timestamp')
        chart_placeholder.line_chart(df['occupancy'])
        time.sleep(0.02)
    cap.release()
    st.session_state.cap = None
else:
    st.warning("Pilih mode dan sambungkan video/webcam.")
