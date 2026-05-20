import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os

# ==========================================
# ⚙️ 核心初始化與配置
# ==========================================
st.set_page_config(
    page_title="🚗 車牌一指辨 — 終極完全體",
    page_icon="🚗",
    layout="wide"
)

# 初始化 EasyOCR Reader（設定支援英文與數字，並啟用 GPU 加速）
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['en'], gpu=True)

# 載入我們大滿貫完賽的 YOLOv8 黃金權重
@st.cache_resource
def load_yolo_model():
    # 請確保路徑正確，如果是本地或 Streamlit Cloud 請對齊你的權重檔案路徑
    model_path = "/content/drive/MyDrive/AI_test/AI_final_CAR/checkpoints_yolo/fold_5/weights/best.pt"
    if not os.path.exists(model_path):
        # 容錯機制：若路徑不存在則嘗試讀取當前目錄下的 best.pt
        model_path = "best.pt"
    return YOLO(model_path)

try:
    reader = load_ocr_reader()
    model = load_yolo_model()
except Exception as e:
    st.error(f"❌ 模型載入失敗，請檢查權重路徑。錯誤訊息: {e}")

# ==========================================
# 📐 幾何與影像處理外掛函式
# ==========================================
def correct_plate_rotation(img):
    """
    OpenCV 幾何轉正物理外掛：
    透過二值化、尋找最大輪廓，並利用最小外接矩形（MinAreaRect）計算傾斜角度，
    最後使用仿射變換（Affine Transformation）將車牌徹底轉正。
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 大津二值化強行分割前景背景
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
            
        # 抓出面積最大的輪廓（即車牌主體）
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        
        # 角度物理校正邏輯
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
            
        if abs(angle) < 0.5:
            return img
            
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img

# ==========================================
# 🎨 Streamlit 前端介面建構
# ==========================================
st.title("🚗 AI 車牌智慧辨識系統")
st.markdown("---")

# 側邊欄配置
st.sidebar.header("🛠️ 系統參數監控")
st.sidebar.success("✅ 左線戰果：YOLOv8 K-Fold (mAP: 97.5%) 運行中")
st.sidebar.success("✅ 右線戰果：EasyOCR 影像流氓預處理已開光")
confidence_threshold = st.sidebar.slider("🎯 YOLOv8 置信度門檻", 0.10, 1.00, 0.45, 0.05)

# 檔案上傳組件
uploaded_file = st.file_uploader("📸 請上傳車輛照片 (支援 JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取並解碼上傳的影像
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.subheader("🖼️ 原始影像與偵測進度")
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="上傳的原始圖片", use_container_width=True)
    
    # 啟動 YOLOv8 推論
    with st.spinner("🚀 YOLOv8 正在精準定位車牌邊界框..."):
        results = model.predict(source=original_image, conf=confidence_threshold, verbose=False)
        
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        st.warning("⚠️ 系統未能偵測到任何車plate目標，請嘗試調低置信度門檻。")
    else:
        st.info(f"🎉 成功偵測到 {len(boxes)} 個車牌目標！開始啟動右線極限解碼...")
        
        # 遍歷所有偵測到的車牌
        for idx, box in enumerate(boxes):
            st.markdown(f"### 📍 車牌目標 #{idx + 1}")
            
            # 取得邊界框座標 (int)
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            
            # 進行車牌物理裁剪 (Crop)
            cropped_plate = original_image[y1:y2, x1:x2]
            
            # 確保裁剪出來的圖片不是空的
            if cropped_plate.size == 0:
                continue
                
            # 1. 啟動 OpenCV 幾何轉正物理外掛
            fixed_plate = correct_plate_rotation(cropped_plate)
            
            # =====================================================================
            # 🔥 右線單兵決戰外掛：CLAHE 局部對比度強化 + 15 像素白框外擴
            # =====================================================================
            try:
                # 步驟 A: 轉為灰階圖（移除色彩雜訊，讓文字結構更純粹）
                gray_plate = cv2.cvtColor(fixed_plate, cv2.COLOR_BGR2GRAY)
                
                # 步驟 B: 啟動 CLAHE 自適應直方圖均衡化
                # clipLimit=3.0 強行拉開局部黑白對比，專治大燈反光死白或環境過暗
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced_plate = clahe.apply(gray_plate)
                
                # 步驟 C: 物理外擴純白邊框（上下左右各加 15 像素）
                # 給 EasyOCR 的滑動視窗留出呼吸空間，徹底根治因「切圖太貼邊」導致的邊緣字體變形漏字
                final_for_ocr = cv2.copyMakeBorder(
                    enhanced_plate, 15, 15, 15,
