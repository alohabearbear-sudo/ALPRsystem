import os
import gc
import sys
import requests
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR  # 💡 替換：改引入 PaddleOCR
from PIL import Image, ImageOps
import contextlib
import logging

# --- 0. Streamlit 網頁基本配置 ---
st.set_page_config(
    page_title="AI智慧車牌辨識系統",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. 設定與路徑 ---
MODEL_URL = "https://github.com/alohabearbear-sudo/Car-Plate-Recognition/releases/download/v1/best.pt"
MODEL_PATH = "best.pt"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        # 💡 因為 CSS 封鎖了 .stSpinner，這裡改用常駐文字提示，避免首次下載時畫面看起來像卡死
        msg = st.empty()
        msg.warning("⏳ 首次啟動，正在下載 YOLO 車牌偵測模型，請稍候...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        msg.empty()
    return YOLO(MODEL_PATH)

@st.cache_resource
def get_ocr():
    # 強力封鎖 paddleocr 與全域 logging 輸出，全面消滅日誌字樣
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            # 💡 建立 PaddleOCR 實例（使用英文模型，關閉方向分類器以加速）
            reader = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
    return reader

# --- 2. 核心辨識邏輯 (完全保留 Jimmy 的中心點重構與 AI 辨識邏輯) ---
def process_recognition(img_np, should_flip=False):
    if img_np is None:
        return None, None, "等待輸入...", "0.00%"
        
    # 由於調用手機原廠相機與相簿檔案方向完全正確，後台不需要再進行任何額外的翻轉，確保定格不反向
    if should_flip:
        pass  
        
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
                cv2.putText(draw_img, "License Plate", (xmin, ymin-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                
                plate_crop = img_np[ymin:ymax, xmin:xmax]
                if plate_crop.size == 0: 
                    continue
                
                plate_crop_res = plate_crop
                
                # 💡 調整：PaddleOCR 對於彩色或灰階影像皆有很好的適應力，此處沿用你的尺寸調整
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        # 💡 替換為 PaddleOCR 辨識語法
                        _result = reader.ocr(resized, cls=False)
                
                # 💡 解析 PaddleOCR 的回傳結構 [ [ [ [box], (text, score) ], ... ] ]
                if _result and _result[0]:
                    words = []
                    scores = []
                    for line in _result[0]:
                        text_found = line[1][0].upper()
                        # 過濾出英數字與減號
                        filtered_text = "".join([c for c in text_found if c.isalnum() or c == '-'])
                        if filtered_text:
                            words.append(filtered_text)
                            scores.append(line[1][1])
                    
                    if words:
                        plate_no_res = " ".join(words)
                        conf_res = f"{np.mean(scores):.2%}"  # 多段文字則取平均信心值
        
        if not found_plate:
            plate_no_res = "❌ 找不到車牌"
            conf_res = "0.00%"
            
    except Exception as e:
        plate_no_res = f"❌ 辨識異常: {str(e)}"
    finally:
        gc.collect()
        
    return draw_img, plate_crop_res, plate_no_res, conf_res

# --- 3. 前端 CSS 強力控制項：全面封鎖網頁載入提示字樣 ---
st.markdown("""
<style>
    .stMarkdown h1 { color: #1E88E5; text-align: center; font-weight: bold; }
    .stMarkdown h3 { text-align: center; color: #555; }
    
    /* 徹底隱藏 Streamlit 的右上角 Running 狀態、旋轉小圖示與任何通知字樣 */
    div[data-testid="stStatusWidget"],
    .stSpinner,
    div[data-testid="stNotification"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 建立 UI 介面 ---
st.markdown("# 🚗 (Beta 版) 車牌一指辨 by Jimmy Chen")
st.markdown("### 🎯 請協助測試並反饋 ")

# 直接常駐單一最高權限上傳與拍照入口
st.markdown("##### 📷 點擊upload選擇【拍照或上傳相簿照片】即可開啟手機後置鏡頭")
upload_file = st.file_uploader("👉 只要一個動作，即可輕鬆辨識車牌", type=["jpg", "jpeg", "png"], key="jimmy_unified_uploader")

st.write("---")

# --- 5. 畫面渲染雙欄架構 ---
col_left, col_right = st.columns([3, 2])

if upload_file is not None:
    # 讀取影像並自動導正手機拍照的 EXIF 旋轉資訊
    raw_img = Image.open(upload_file)
    fixed_img = ImageOps.exif_transpose(raw_img).convert('RGB')
    
    if fixed_img.width > 1024:
        fixed_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    img_np = np.array(fixed_img)
    
    # 照片直接送入核心 AI 辨識，方向絕對與手機拍到的一模一樣！
    res_draw, res_crop, res_text, res_conf = process_recognition(img_np, should_flip=False)
    
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
