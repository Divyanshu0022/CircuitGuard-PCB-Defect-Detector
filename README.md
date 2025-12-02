# CircuitGuard: PCB Defect Detection System (YOLOv8)

**CircuitGuard** is an automated computer vision system designed to detect and classify defects in Printed Circuit Boards (PCBs). The system utilizes **State-of-the-art Object Detection (YOLOv8)** to localize tiny defects such as missing holes and short circuits with high precision.

> **🎓 Academic Context:**
> This project was developed as a key deliverable for the **Infosys Springboard Internship**. It represents the practical application of Deep Learning and AI concepts taught during the **8-week internship program**.

## 🚀 Project Status
Current Focus: **Phase 2 - Model Training & Inference**

| Milestone | Module | Task Description | Status |
| :--- | :--- | :--- | :--- |
| **Milestone 1** | **Module 1** | **Dataset Preparation** <br> (XML to YOLO conversion, Folder restructuring, Class Mapping) | ✅ **Done** |
| **Milestone 2** | **Module 2** | **Model Training (YOLOv8)** <br> (Transfer Learning, High-Res Training @ 1024px) | ✅ **Done** |
| **Milestone 3** | **Module 3** | **Inference & Validation** <br> (Bounding box generation, Confidence scoring, Model Saving) | ✅ **Done** |
| **Milestone 4** | Module 4 | Frontend Web UI (Streamlit/HTML) | ⏳ *Pending* |
| | Module 5 | Deployment & Backend Integration | ⏳ *Pending* |

---

## 📂 Project Structure & Setup
This project is designed to run on **Google Colab** (utilizing T4 GPU) with data persistence via **Google Drive**.

### **1. The Dataset**
The project uses the **DeepPCB Dataset**, formatted for YOLO.
* **Format:** YOLO v8 (TXT labels with normalized coordinates).
* **Classes:** `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spur`, `Spurious_copper`.

### **2. Environment Setup**
The notebook handles the transition from raw data to a trained model.

**To run this code:**
1.  Upload `PCB_DATASET.zip` to Google Drive:
    ```text
    /My Drive/DeepPCB_Project/PCB_DATASET.zip
    ```
2.  Open `PCB_Defect_detection_with_yolo_final.ipynb` in Colab.
3.  Mount Drive; the script auto-extracts and converts data to the YOLO directory structure:
    ```text
    /content/yolo_dataset/
    ├── train/
    ├── val/
    └── test/
    ```

---

## 🛠️ How It Works (YOLOv8 Workflow)
<img width="898" height="735" alt="image" src="https://github.com/user-attachments/assets/c681b220-53e9-47cf-9a5d-34156147b61d" />


### **Step 1: Robust Data Preprocessing**
Instead of simple image subtraction, we prepare data for a Neural Network:
* **Path Correction:** Auto-detection of image paths regardless of folder depth.
* **Label Conversion:** Parses Pascal VOC (XML) annotations.
* **Smart Mapping:** Handles case-sensitivity issues (e.g., mapping `missing_hole` -> `Missing_hole`) to ensure zero data loss.

### **Step 2: Model Training**
We utilize **Ultralytics YOLOv8 (Small version)**.
* **Model:** `yolov8s.pt` (Small) - chosen for the balance between speed and small-object detection capabilities.
* **Resolution:** Trained at **1024x1024** pixels. High resolution is critical for detecting tiny PCB defects like "Missing Holes" that disappear at standard resolutions (640px).
* **Epochs:** 25 Epochs with Early Stopping patience.

### **Step 3: Inference**
The trained model predicts defects on unseen test images:
* **Output:** Draws Bounding Boxes around defects.
* **Labels:** Assigns class names and confidence scores (e.g., `Open_circuit 0.85`).
* **Persistence:** The best model weights (`best.pt`) are automatically saved to Google Drive for future use.

---

## 📸 Output Examples
The system outputs high-resolution images with color-coded bounding boxes identifying specific defects.

* **Input:** Raw PCB Image.
* **Output:** Annotated Image with defects localized + Confidence Score.

---

## 🧰 Tech Stack
* **Deep Learning Framework:** PyTorch
* **Object Detection:** Ultralytics YOLOv8
* **Language:** Python 3.10+
* **Image Processing:** OpenCV (`cv2`)
* **Environment:** Google Colab (GPU Accelerated)

---

## 👨‍💻 Future Work
The current model achieves an mAP (Mean Average Precision) suitable for demonstration. Next steps involve:
1.  Building a **Streamlit Web Interface** to allow users to upload a PCB image and see defects instantly.
2.  Optimizing the model for edge deployment (ONNX export).
