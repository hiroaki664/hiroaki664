## Hi there 👋

# 🧪 Nanoparticle Segmentation and Defect Detection from SEM Images

This project focuses on detecting and measuring nanoparticles in SEM images  
using deep learning models (YOLOSeg, U-Net) and classical filters (DoG, NLM).  
In addition, we plan to extend the workflow to **defect and impurity detection**  
using **Region-based Convolutional Neural Networks (R-CNN)** to distinguish  
particles from attached contaminants and foreign materials.

---

## 📋 Pipeline
1. **Preprocessing** – Flat-field correction, DoG, Non-local Means  
2. **Segmentation** – YOLOSeg for nanoparticle extraction  
3. **Postprocessing** – Contour analysis and particle sizing  
4. **Defect Detection (in progress)** – R-CNN-based impurity and defect classification

---

## 🧠 Results
- Mean IoU: 0.73  
- AP: 0.82  
- Robust performance for < 500 nm particles  
- Ongoing development for adaptive impurity classification based on material type

---

## 🖼 Example
![Example result](results/overlay_sample.png)

