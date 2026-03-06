# Eye Disease Classification using Color Fundus Images

This project implements deep learning models to automatically classify retinal eye diseases from **color fundus images**. The goal is to assist early diagnosis of ophthalmic conditions using computer vision and modern deep learning architectures.

The system evaluates multiple state-of-the-art models including **DenseNet201, EfficientNet-B4, Vision Transformer (ViT), and a Hybrid CNN–Transformer model** for multi-class retinal disease classification. 

---

## Overview

Retinal diseases such as **diabetic retinopathy, glaucoma, and macular degeneration** are major causes of visual impairment worldwide. Early detection is critical, but manual examination of fundus images requires expert ophthalmologists and can be time-consuming.

This project explores deep learning approaches to automatically classify retinal diseases using color fundus images, improving screening efficiency and diagnostic support. 

---

## Dataset

The models were trained and evaluated using the **Eye Disease Image Dataset**, which contains:

* **21,577 fundus images**
* **10 eye disease categories**
* **5,335 original images + augmented samples**

Example disease classes include:

* Diabetic Retinopathy
* Glaucoma
* Disc Edema
* Retinal Detachment
* Pterygium
* Myopia
* Macular Scar
* Retinitis Pigmentosa
* Central Serous Chorioretinopathy
* Healthy

---

## Methodology

The overall workflow of the project is:

1. **Image Preprocessing**

   * Resize images to 224 × 224
   * Normalize pixel values
   * Noise reduction
   * Contrast enhancement

2. **Data Augmentation**

   * Random rotations
   * Horizontal flips
   * Zoom and brightness adjustments

3. **Model Training**

The following deep learning architectures were implemented and compared:

* DenseNet201
* EfficientNet-B4
* Vision Transformer (ViT)
* Hybrid CNN–Transformer Model

4. **Evaluation Metrics**

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Five-fold cross validation was used to ensure robust performance evaluation.

---

## Results

| Model                    | Accuracy   |
| ------------------------ | ---------- |
| EfficientNet-B4          | **90.49%** |
| DenseNet201              | 88.45%     |
| Hybrid CNN–Transformer   | 88.15%     |
| Vision Transformer (ViT) | 85.22%     |

EfficientNet-B4 achieved the best overall performance for retinal disease classification.

---

## Technologies Used

* Python
* TensorFlow / PyTorch
* OpenCV
* NumPy
* Scikit-learn

---

## Applications

This system can support:

* Automated retinal disease screening
* AI-assisted ophthalmic diagnostics
* Clinical decision support systems
* Telemedicine and remote healthcare



## Future Work

Future improvements may include:

* Training on multi-hospital datasets
* Improving model interpretability (Grad-CAM, attention maps)
* Deployment in real-time clinical screening systems

---

## License

This project is intended for **research and educational purposes**.
