# AI Face Attendance System

This project is an AI-based face recognition attendance system.

The system detects student faces from a classroom image or camera capture and automatically marks attendance.

## Features

- Face recognition using InsightFace
- Upload classroom image
- Capture classroom image from camera
- Automatic attendance marking
- Save attendance to CSV file

## Technologies Used

- Python
- Streamlit
- InsightFace
- OpenCV
- NumPy
- Pandas

## Project Workflow

1. Load student images from dataset folder
2. Generate face embeddings
3. Upload or capture classroom image
4. Detect faces
5. Compare embeddings
6. Mark attendance

## Project Structure
AI_Based_Attendance_System
│
├── dataset
│ ├── student1.jpg
│ ├── student2.jpg

├── app.py
└── attendance.csv

## Run the Project

```bash
python -m streamlit run app.py 

Then open:

http://localhost:8501

Author

Balaram Krishna


---

# Step 3 — Save the file

Save `README.md`.

---

# Step 4 — Upload README to GitHub

Run these commands:

```bash
git add README.md 
git push