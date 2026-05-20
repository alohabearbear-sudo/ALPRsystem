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
    model_path = "/content/drive/MyDrive/AI_test/AI_final_CAR/checkpoints_yolo/fold_5/weights/best.pt"
    if not os.path.exists(model_path):
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
    透過最小外接矩形計算傾斜角度，並利用仿射變換將車牌徹底轉正。
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
            
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        
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
st.title("🚗 車牌影像智慧辨識系統 — 五折大滿貫完全體")
st.markdown("---")

# 側邊欄配置
st.sidebar.header("🛠️ 系統參數監控")
st.sidebar.success("✅ 左線戰果：YOLOv8 K-Fold (mAP: 97.5%) 運行中")
st.sidebar.success("✅ 右線戰果：EasyOCR 影像流氓預處理已開光")
confidence_threshold = st.sidebar.slider("🎯 YOLOv8 置信度門檻", 0.10, 1.00, 0.45, 0.05)

# 檔案上傳組件
uploaded_file = st.file_uploader("📸 請上傳車輛照片 (支援 JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.subheader("🖼️ 原始影像與偵測進度")
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="上傳的原始圖片", use_container_width=True)
    
    # 啟動 YOLOv8 推論
    with st.spinner("🚀 YOLOv8 正在精準定位車牌邊界框..."):
        results = model.predict(source=original_image, conf=confidence_threshold, verbose=False)
        
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        st.warning("⚠️ 系統未能偵測到任何車牌目標，請嘗試調低置信度門檻。")
    else:
        st.info(f"🎉 成功偵測到 {len(boxes)} 個車牌目標！開始啟動右線極限解碼...")
        
        for idx, box in enumerate(boxes):
            st.markdown(f"### 📍 車牌目標 #{idx + 1}")
            
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x
