import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 1. เก็บข้อมูลเลข 2 หลักใน Session State
if 'data' not in st.session_state:
    st.session_state.data = []  # เก็บตัวเลขที่กรอก

# รายชื่อเลขที่มนุษย์ระบุว่าไม่น่าจะออก
excluded_numbers = [13, 47, 88]  # ตัวอย่างเลข

# 2. ฟังก์ชันสุ่มเลข 2 หลักในรูปแบบสตริง 2 หลักเสมอ
def generate_random_number():
    random_number = np.random.randint(0, 100)  # สุ่มเลข 0-99
    return f"{random_number:02d}"  # แปลงเป็นสตริง 2 หลัก เช่น "01", "09"

# 3. ฟังก์ชันทำนายผลเลขหลักเดียว 5 ตัว
def predict_top_5_single_digits(data, n_samples):
    if len(data) < n_samples + 1:  # ต้องมีข้อมูลเพียงพอ
        return ["ไม่เพียงพอสำหรับการทำนาย"]

    # เตรียมข้อมูล: X คือข้อมูล, y คือตัวถัดมา
    X = np.array(data[-(n_samples + 1):-1]).reshape(-1, 1)
    y = np.array(data[-n_samples:])

    # ปรับจำนวน neighbors ให้เหมาะกับข้อมูลที่มี
    n_neighbors = min(5, len(X))

    # สร้างโมเดล KNN และฝึกโมเดล
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)

    # ใช้ข้อมูลตัวล่าสุดเพื่อทำนาย
    latest_number = np.array([data[-1]]).reshape(1, -1)

    # หาเพื่อนบ้านที่ใกล้ที่สุดและดึงเลขที่คาดการณ์ได้
    neighbors = model.kneighbors(latest_number, return_distance=False)
    predictions = y[neighbors[0]]

    # กรองเลขที่ไม่น่าจะออก
    filtered_predictions = [num for num in predictions if num not in excluded_numbers]

    # แปลงเลขสองหลักเป็นหลักเดียว
    single_digits = [int(str(num).zfill(2)[i]) for num in filtered_predictions for i in range(2)]

    # นับความถี่ของแต่ละหลัก และเลือก 5 อันดับที่มีความถี่สูงสุด
    unique, counts = np.unique(single_digits, return_counts=True)
    top_5_digits = [str(digit) for digit in unique[np.argsort(-counts)][:5]]

    return top_5_digits

# ส่วนของ UI
st.title("AI ทำนายเลขหลักเดียว (พร้อมกฎมนุษย์)")

# 4. รับข้อมูลจากผู้ใช้
user_input = st.text_input("กรอกเลข 2 หลัก (00-99):")

if st.button("บันทึกข้อมูล"):
    if user_input.isdigit() and 0 <= int(user_input) <= 99:
        st.session_state.data.append(int(user_input))
        st.success(f"บันทึกเลข {user_input.zfill(2)} เรียบร้อยแล้ว!")
    else:
        st.error("กรุณากรอกเลข 2 หลักระหว่าง 00 ถึง 99")

# 5. ปุ่มสุ่มเลข 2 หลัก
if st.button("สุ่มเลข 2 หลัก"):
    random_number = generate_random_number()
    st.session_state.data.append(int(random_number))
    st.info(f"สุ่มได้เลข: {random_number}")

# 6. Slider เพื่อเลือกจำนวนข้อมูลที่ใช้ในการทำนาย
n_samples = st.slider("เลือกจำนวนข้อมูลที่ต้องการใช้ในการทำนาย", min_value=2, max_value=min(30, len(st.session_state.data)), value=5)

# 7. ปุ่มทำนายผล
if st.button("ทายผลเลขถัดไป"):
    if len(st.session_state.data) < n_samples + 1:
        st.warning("กรุณาเพิ่มข้อมูลให้เพียงพอสำหรับการทำนาย")
    else:
        top_5_digits = predict_top_5_single_digits(st.session_state.data, n_samples)
        st.success(f"เลขหลักเดียวที่มีแนวโน้มจะออก (5 อันดับ): {', '.join(top_5_digits)}")

# 8. แสดงข้อมูลที่เก็บไว้
st.subheader("ข้อมูลเลข 2 หลักที่เก็บไว้:")
st.write([f"{num:02d}" for num in st.session_state.data])
