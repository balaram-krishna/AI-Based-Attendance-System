# Import required libraries
import os, cv2, numpy as np, pandas as pd, streamlit as st
from insightface.app import FaceAnalysis
from PIL import Image, ImageDraw
from datetime import datetime

# Configuration variables
DATASET = "dataset"          # Folder where student images are stored
CSV_FILE = "attendance.csv"  # File where attendance will be saved
THRESHOLD = 0.60             # Similarity threshold for recognition

# Configure Streamlit page
st.set_page_config(page_title="Face Attendance", layout="wide")

# Create dataset folder if it doesn't exist
os.makedirs(DATASET, exist_ok=True)

# ------------------ Load Face Recognition Model ------------------

@st.cache_resource
def load_model():
    # Load InsightFace pretrained model (buffalo_l)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640,640))  # CPU mode
    return app

model = load_model()

# ------------------ Load Student Images ------------------

def load_students():
    data = {}

    # Read every image inside dataset folder
    for f in os.listdir(DATASET):

        img = cv2.imread(os.path.join(DATASET,f))

        # Detect faces in the image
        faces = model.get(img) if img is not None else []

        # If face found, store its embedding
        if faces:
            roll = os.path.splitext(f)[0]   # filename becomes roll number
            data[roll] = faces[0].normed_embedding

    return data

# Load all students
students = load_students()

# ------------------ Streamlit UI ------------------

st.title("📸 AI Face Attendance System")

# Show number of students loaded
st.write("Students Loaded:", len(students))

# ------------------ Upload Classroom Image ------------------

upload = st.file_uploader(
    "Upload Classroom Photo",
    type=["jpg","jpeg","png"]
)

# ------------------ Camera Capture Section ------------------

# Session state keeps camera images
if "cams" not in st.session_state:
    st.session_state.cams = []

# Buttons to add or remove camera capture
c1,c2 = st.columns(2)

if c1.button("➕ Add Capture") and len(st.session_state.cams) < 3:
    st.session_state.cams.append(None)

if c2.button("➖ Remove Capture") and st.session_state.cams:
    st.session_state.cams.pop()

# Camera input widgets
for i in range(len(st.session_state.cams)):
    cam = st.camera_input(f"Capture {i+1}", key=f"cam{i}")
    if cam:
        st.session_state.cams[i] = cam.read()

# ------------------ Collect Images ------------------

images = []

# If user uploaded image
if upload:
    images.append(upload.read())

# If user captured images
for c in st.session_state.cams:
    if c:
        images.append(c)

# ------------------ Process Attendance ------------------

if st.button("Process Attendance"):

    # Ensure at least one image is provided
    if not images:
        st.error("Upload or capture at least 1 photo")
        st.stop()

    recognised = set()

    # Process each image
    for data in images:

        # Convert bytes to image
        img = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)

        # Detect faces
        faces = model.get(img)

        # Convert image for drawing boxes
        view = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(view)

        for face in faces:

            emb = face.normed_embedding
            roll, best = None, -1

            # Compare detected face with stored embeddings
            for r,e in students.items():

                score = np.dot(e, emb)

                if score > best:
                    best, roll = score, r

            x1,y1,x2,y2 = map(int,face.bbox)

            # If similarity passes threshold → recognized
            if best > THRESHOLD:

                recognised.add(roll)
                color=(0,200,0)
                label=roll

            else:
                color=(255,0,0)
                label="Unknown"

            # Draw rectangle around face
            draw.rectangle([x1,y1,x2,y2],outline=color,width=3)
            draw.text((x1,y1-10),label,fill=color)

        # Show processed image
        st.image(view,width="stretch")

    # ------------------ Attendance Table ------------------

    attendance = [{
        "Roll": r,
        "Status": "Present" if r in recognised else "Absent",
        "Time": datetime.now().strftime("%H:%M:%S")
    } for r in students]

    df = pd.DataFrame(attendance)

    st.subheader("Attendance")
    st.dataframe(df)

    # ------------------ Save Attendance ------------------

    if st.button("Save Attendance"):

        old = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()

        pd.concat([old,df]).to_csv(CSV_FILE,index=False)

        st.success("Attendance saved")