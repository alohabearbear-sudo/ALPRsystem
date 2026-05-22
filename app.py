import os
import gc
import sys
import traceback
import requests
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import easyocr
from PIL import Image, ImageOps

# --- 0. Streamlit 網頁基本配置 ---
st.set_page_config(
    page_title="AI智慧車牌辨識系統",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. 設定與路徑 ---
MODEL_URL = "https://github.com/alohabearbear-sudo/Car-Plate-Recognition/releases/download/v1/best.pt"
MODEL_PATH = "best.pt"

# --- 全域模型快取（不用 @st.cache_resource 避免快取到損壞模型）---
_model = None
_reader = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO("best.pt")
    return _model

def get_ocr():
    global _reader
    if _reader is None:
        with st.spinner("⏳ 正在載入模型..."):
            _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

# --- 2. 核心辨識邏輯 ---
def process_recognition(img_np):
    if img_np is None:
        return None, None, "等待輸入...", "0.00%"

    h, w = img_np.shape[:2]
    draw_img = img_np.copy()

    plate_crop_res = None
    plate_no_res = "❌ 找不到車牌"
    conf_res = "0.00%"

    try:
        model = get_model()
        reader = get_ocr()

        results = model.predict(img_np, conf=0.4, verbose=False)
        found_plate = False

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                found_plate = True
                x1, y1, x2, y2 = map(int, box)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                original_max_side = max(x2 - x1, y2 - y1)
                new_w_half = int(original_max_side * 0.6)
                new_h_half = int(original_max_side * 0.25)

                xmin, xmax = max(0, cx - new_w_half), min(w, cx + new_w_half)
                ymin, ymax = max(0, cy - new_h_half), min(h, cy + new_h_half)

                cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 6)
                cv2.putText(draw_img, "License Plate", (xmin, ymin - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

                plate_crop = img_np[ymin:ymax, xmin:xmax]
                if plate_crop.size == 0:
                    continue

                plate_crop_res = plate_crop

                gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

                _result = reader.readtext(resized, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')

                if _result:
                    words = []
                    scores = []
                    for bbox, text_found, prob in _result:
                        text_upper = text_found.upper()
                        filtered_text = "".join([c for c in text_upper if c.isalnum() or c == '-'])
                        if filtered_text:
                            words.append(filtered_text)
                            scores.append(prob)

                    if words:
                        plate_no_res = words[-1]
                        conf_res = f"{np.mean(scores):.2%}"

        if not found_plate:
            plate_no_res = "❌ 找不到車牌"
            conf_res = "0.00%"

    except Exception as e:
        plate_no_res = f"❌ 辨識異常: {str(e)}"
        print("=== TRACEBACK ===")
        print(traceback.format_exc())
        print("=== END ===")
    finally:
        gc.collect()

    return draw_img, plate_crop_res, plate_no_res, conf_res

# --- 3. 前端 CSS ---
st.markdown("""
<style>
    .stMarkdown h1 { color: #1E88E5; text-align: center; font-weight: bold; }
    .stMarkdown h3 { text-align: center; color: #555; }
</style>
""", unsafe_allow_html=True)

# --- 4. 建立 UI 介面 ---
st.markdown("# 🚗 (Beta 版) 車牌一指辨 by Jimmy Chen")
st.markdown("### 🎯 採用 YOLOv8 + easyOCR 架構")
st.markdown("##### 📷 點擊按鈕【拍照或上傳相簿照片】即可開啟手機後置鏡頭")

upload_file = st.file_uploader(
    "👉 只要一個動作，即可輕鬆辨識車牌",
    type=["jpg", "jpeg", "png"],
    key="uploader_v3"
)

st.write("---")

# --- 5. 畫面渲染 ---
col_left, col_right = st.columns([3, 2])

if upload_file is not None:
    st.write(f"✅ 已收到圖片：{upload_file.name}，大小：{upload_file.size} bytes")

    try:
        upload_file.seek(0)
        raw_img = Image.open(upload_file)
        raw_img.load()
    except Exception as e:
        st.error(f"圖片讀取失敗: {e}")
        print(traceback.format_exc())
        st.stop()

    try:
        fixed_img = ImageOps.exif_transpose(raw_img).convert('RGB')
    except Exception:
        fixed_img = raw_img.convert('RGB')

    if fixed_img.width > 1024:
        fixed_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    img_np = np.array(fixed_img)

    with st.spinner("🔍 辨識中，請稍候..."):
        res_draw, res_crop, res_text, res_conf = process_recognition(img_np)

    with col_left:
        st.subheader("1. 定位確認")
        if res_draw is not None:
            st.image(res_draw, use_container_width=True)

    with col_right:
        st.subheader("2. 水平精準裁切區域")
        if res_crop is not None:
            st.image(res_crop, use_container_width=True)
        st.subheader("🔢 辨識號碼")
        st.info(f"**{res_text}**")
        st.subheader("信心值")
        st.metric(label="Confidence", value=res_conf)
else:
    gc.collect()
    with col_left:
        st.info("💡 請點擊上方按鈕，選擇【拍照】或從【上傳相簿照片】來啟動車牌自動辨識。")
    with col_right:
        st.text_input("🔢 辨識號碼", value="等待輸入...", disabled=True, key="disabled_output_box")

st.markdown("---")
st.markdown("<center>Developed by Jimmy Chen | 2026 High-Performance Native Version</center>", unsafe_allow_html=True)
