import os
import gc
import sys
import requests
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR  # 💡 正統 PaddleOCR 引用
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
            # 💡 終極修正：拔掉 use_gpu 參數，只保留最核心的 lang='en'，讓系統自動適應環境
            reader = PaddleOCR(use_angle_cls=False, lang='en')
    return reader

# --- 2. 核心辨識邏輯 (完全保留 Jimmy 的中心點重構與 AI 辨識邏輯) ---
def process_recognition(img_np, should_flip=False):
    if img_np is None:
        return None, None, "等待輸入...", "0.00%"
        
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
                
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        _result = reader.ocr(resized, cls=False)
                
                # 💡 解析正統 PaddleOCR 結構 [ [ [ [box], (text, score) ], ... ] ]
                if _result and _result[0]:
                    words = []
                    scores = []
                    for line in _result[0]:
                        text_found = line[1][0].upper()
                        filtered_text = "".join([c for c in text_found if c.isalnum() or c == '-'])
                        if filtered_text:
                            words.append(filtered_text)
                            scores.append(line[1][1])
                    
                    if words:
                        plate_no_res = " ".join(words)
                        conf_res = f"{np.mean(scores):.2%}"
        
        if not found_plate:
            plate_no_res = "❌ 找不到車牌"
            conf_res = "0.00%"
            
    except Exception as e:
        plate_no_res = f"❌ 辨識異常: {str(e)}"
    finally:
        gc.collect()
        
    return draw_img, plate_crop_res, plate_no_res, conf_res
