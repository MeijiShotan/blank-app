import streamlit as st
import requests
import io
import base64
from PIL import Image, ImageDraw, ImageOps
# ตั้งค่า API ของ Roboflow


ROBOFLOW_API_KEY = "9BCXeL5a6Vgvn8eqPSR8"
PROJECT_ID = "anemia_pcm"
MODEL_VERSION = "2025-03-31 11:26pm"
API_URL = "https://outline.roboflow.com/anemia_pcm/2?api_key=9BCXeL5a6Vgvn8eqPSR8"

st.title("Image Segmentation Viewer")
st.write("อัปโหลดรูปภาพเพื่อดูผลลัพธ์ของโมเดล")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)

    # ลดขนาดภาพเพื่อให้ API รองรับ
    max_size = (512, 512)
    image.thumbnail(max_size)

    with st.spinner("กำลังประมวลผล..."):
        try:
            # แปลงภาพเป็น base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode()

            # ส่งไปยัง API
            response = requests.post(API_URL, files={"file": ("image.jpg", io.BytesIO(buffered.getvalue()), "image/jpeg")})
            st.write(f"Response Status Code: {response.status_code}")
            st.write(f"Response JSON: {response.text}")

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])

                if predictions:
                    # สร้าง mask เปล่า
                    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))  # ใช้ RGBA สำหรับ transparency

                    # วาด mask บนพื้นฐานของ points ใน predictions
                    for pred in predictions:
                        points = pred["points"]
                        polygon = [(point["x"], point["y"]) for point in points]
                        draw = ImageDraw.Draw(mask)
                        draw.polygon(polygon, fill=(0, 255, 0, 128))  # วาดเส้น polygon สีเขียวที่มีความโปร่งใส

                    # Overlay mask บนภาพต้นฉบับ
                    result_image = Image.alpha_composite(image.convert("RGBA"), mask)  # แปลงภาพให้เป็น RGBA แล้วรวมกับ mask

                    st.image(result_image, caption="ผลลัพธ์จากโมเดล", use_column_width=True)
                else:
                    st.write("API ไม่พบวัตถุในภาพ หรือไม่ได้ส่ง mask กลับมา")
            else:
                st.write("เกิดข้อผิดพลาดในการประมวลผล:", response.text)
        except Exception as e:
            st.write(f"เกิดข้อผิดพลาด: {e}")
