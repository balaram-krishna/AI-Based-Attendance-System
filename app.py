import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw
from datetime import datetime

# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "Dataset")
CSV_FILE = os.path.join(BASE_DIR, "attendance.csv")
THRESHOLD = 0.60

st.set_page_config(page_title="AI Face Attendance System", layout="wide")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640,640))
    return app

model = load_model()

# ---------------- LOAD STUDENTS ----------------

def load_students():
    data = {}

    if not os.path.exists(DATASET):
        return data

    for f in os.listdir(DATASET):

        path = os.path.join(DATASET, f)

        img = cv2.imread(path)

        if img is None:
            continue

        faces = model.get(img)

        if faces:
            roll = os.path.splitext(f)[0]
            data[roll] = faces[0].normed_embedding

    return data


students = load_students()

# ---------------- UI ----------------

st.title("📸 AI Face Attendance System")

st.write("Students Loaded:", len(students))

# ---------------- IMAGE UPLOAD ----------------

upload = st.file_uploader(
    "Upload Classroom Photo",
    type=["jpg","jpeg","png"]
)

# ---------------- CAMERA CAPTURE ----------------

if "cams" not in st.session_state:
    st.session_state.cams = []

col1, col2 = st.columns(2)

if col1.button("➕ Add Capture") and len(st.session_state.cams) < 3:
    st.session_state.cams.append(None)

if col2.button("➖ Remove Capture") and st.session_state.cams:
    st.session_state.cams.pop()

for i in range(len(st.session_state.cams)):
    cam = st.camera_input(f"Capture {i+1}", key=f"cam{i}")

    if cam:
        st.session_state.cams[i] = cam.read()

# ---------------- COLLECT IMAGES ----------------

images = []

if upload:
    images.append(upload.read())

for c in st.session_state.cams:
    if c:
        images.append(c)

# ---------------- PROCESS ATTENDANCE ----------------

if st.button("Process Attendance"):

    if not images:
        st.warning("Upload or capture at least 1 photo")
        st.stop()

    recognised = set()

    for data in images:

        img = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)

        faces = model.get(img)

        view = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(view)

        for face in faces:

            emb = face.normed_embedding
            roll = None
            best = -1

            for r,e in students.items():

                score = np.dot(e,emb)

                if score > best:
                    best = score
                    roll = r

            x1,y1,x2,y2 = map(int,face.bbox)

            if best > THRESHOLD:

                recognised.add(roll)
                color = (0,200,0)
                label = roll

            else:
                color = (255,0,0)
                label = "Unknown"

            draw.rectangle([x1,y1,x2,y2],outline=color,width=3)
            draw.text((x1,y1-10),label,fill=color)

        st.image(view,width="stretch")

    # ---------------- ATTENDANCE TABLE ----------------

    attendance = []

    for r in students:

        status = "Present" if r in recognised else "Absent"

        attendance.append({
            "Roll": r,
            "Status": status,
            "Time": datetime.now().strftime("%H:%M:%S")
        })

    df = pd.DataFrame(attendance)

    st.subheader("Attendance")
    st.dataframe(df)

    # ---------------- SAVE ATTENDANCE ----------------

    if st.button("Save Attendance"):

        if os.path.exists(CSV_FILE):
            old = pd.read_csv(CSV_FILE)
            df = pd.concat([old,df])

        df.to_csv(CSV_FILE,index=False)

        st.success("Attendance saved successfully")