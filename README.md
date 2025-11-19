# 🛡️ SafetyLens: Active Learning MLOps Pipeline

**SafetyLens** is an end-to-end MLOps computer vision system designed for industrial safety monitoring. Unlike standard detection scripts, this project implements an **Active Learning Loop**: it serves predictions via a FastAPI microservice and automatically captures "low-confidence" data to a Data Lake for future retraining.

## 🏗️ Architecture


The system consists of three core components:
1.  **Inference Engine:** A **FastAPI** microservice wrapping a **YOLOv8** model.
2.  **Active Learning Trigger:** A logic layer that filters predictions. If confidence < 40%, the raw image is isolated and saved to a simulated Data Lake (S3) for annotation.
3.  **Frontend:** A **Streamlit** dashboard for real-time testing and visualization.

## 🛠️ Tech Stack
* **ML Framework:** YOLOv8 (Ultralytics), PyTorch
* **Backend:** FastAPI, Uvicorn
* **Data Engineering:** Automated data collection triggers (Simulated ETL)
* **Containerization:** Docker (configured)
* **Frontend:** Streamlit

## 🚀 Installation & Setup

### Prerequisites
* Python 3.9+
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/gnikhilchand/SafetyLens-An-Active-Learning-Object-Detection-System..git
cd safety_lens
