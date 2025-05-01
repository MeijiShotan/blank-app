import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tflite_runtime.interpreter as tflite  
import io
import requests
# ตั้งค่า API ของ Roboflow


ROBOFLOW_API_KEY = "9BCXeL5a6Vgvn8eqPSR8"
PROJECT_ID = "anemia_pcm"
MODEL_VERSION = "2025-03-31 11:26pm"
API_URL = "https://outline.roboflow.com/anemia_pcm/2?api_key=9BCXeL5a6Vgvn8eqPSR8"

st.title("Palpebral conjunctiva detecter")
st.write("อัปโหลดรูปภาพเพื่อดูผลลัพธ์ของโมเดล")
st.write("โดย ธรรญธร ไชยกายุต")
def load_model():
    interpreter = tflite.Interpreter(model_path="M2.tflite")
    interpreter.allocate_tensors()
    return interpreter


def run_segmentation(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # e.g., [1, 224, 224, 3]
    target_size = (input_shape[2], input_shape[1])

    # Resize and normalize image
    img_resized = image.resize(target_size)
    input_data = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assume output is [1, H, W, 1] with values from 0-1 (binary mask)
    mask = output_data[0, ..., 0]
    mask = (mask > 0.5).astype(np.uint8) * 255

    return Image.fromarray(mask).resize(image.size)

# === Upload Image ===
uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])
try:
    with open("M2.tflite", "rb") as f:
        st.success("โหลดไฟล์ .tflite สำเร็จ")
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")

try:
    interpreter = tflite.Interpreter(model_path="M2.tflite")
    interpreter.allocate_tensors()
    st.success("โหลดสำเร็จ")
except Exception as e:
    st.error(f"โหลดไม่สำเร็จ: {e}")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size
    if width > height:
        image = image.rotate(-90, expand=True)

    st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)

    with st.spinner("กำลังประมวลผล..."):
        try:
            interpreter = load_model()
            mask = run_segmentation(image, interpreter)

            # สร้าง RGBA mask สีเขียวโปร่งใส
            green_mask = Image.new("RGBA", image.size, (0, 255, 0, 0))
            mask_data = mask.convert("L").point(lambda x: 128 if x > 0 else 0)
            green_mask.putalpha(mask_data)

            # ซ้อนภาพ
            result_image = Image.alpha_composite(image.convert("RGBA"), green_mask)

            st.image(result_image, caption="ผลลัพธ์จากโมเดล", use_column_width=True)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
