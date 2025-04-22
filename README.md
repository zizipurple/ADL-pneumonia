# Chest X-Ray Pneumonia Detection

This project applies **transfer learning** with multiple pre-trained deep learning models — **VGG16**, **ResNet**, **EfficientNet**, and **CheXNet** — to classify chest X-ray images as either **Normal** or **Pneumonia**.

## 📁 Project Structure

```
chest_xray_classification/
├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
│
├── notebooks/                       
│   ├── vgg16_chest_xray.ipynb
│   ├── resnet_chest_xray.ipynb
│   ├── efficientnet_chest_xray.ipynb
│   └── chexnet_chest_xray.ipynb
│
├── best_vgg16_chest_xray.pth
├── best_resnet_chest_xray.pth
├── best_efficientnet_chest_xray.pth
├── best_chexnet_chest_xray.pth
│
├── README.md
└── requirements.txt
```

## 📌 Models Implemented

| Model         | Base Architecture  | Dataset Used      | Transfer Learning | Final Layer Modifications |
|---------------|--------------------|-------------------|-------------------|----------------------------|
| VGG16         | `torchvision.models.vgg16` | chest_xray (Kermany et al.) | Yes ✅         | Fully Connected + Sigmoid |
| ResNet        | `torchvision.models.resnet18` / `resnet50` | Same | Yes ✅ | Replaced final FC layer |
| EfficientNet  | `efficientnet_b0` from torchvision | Same | Yes ✅ | Custom classifier |
| CheXNet       | `densenet121` pre-trained on ChestX-ray14 | Same | Yes ✅ | Custom classifier |


## 📦 Setup Instructions

1. **Clone the repository:**

git clone https://github.com/your-username/chest-xray-classification.git
cd chest-xray-classification

2. **Create a virtual environment (optional but recommended):**

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install required packages:**

pip install -r requirements.txt

If requirements.txt is not available, install packages manually:

pip install torch torchvision matplotlib seaborn scikit-learn

4. **Download and organize the dataset:**

Download the dataset from Kaggle - Chest X-Ray Images (Pneumonia)
Unzip it and place it as follows:
```
chest_xray/
├── train/
├── val/
└── test/
```
Each subdirectory should contain two folders: NORMAL and PNEUMONIA.

⸻

🚀 Running the Notebooks

Each notebook is self-contained and follows this general structure:
	1.	Import Libraries
	2.	Load and Preprocess Data
	3.	Define Model Architecture
	4.	Train the Model
	5.	Evaluate on Validation and Test Set
	6.	Visualize Results (Loss, Accuracy, Confusion Matrix, Predictions)

⸻

🔧 Launch Jupyter Notebook

Run:

jupyter notebook

Then open one of the following notebooks:
	•	notebooks/vgg16_chest_xray.ipynb
	•	notebooks/resnet_chest_xray.ipynb
	•	notebooks/efficientnet_chest_xray.ipynb
	•	notebooks/chexnet_chest_xray.ipynb

You can modify num_epochs, batch_size, and learning rate inside the notebook as needed.

⸻

🧠 Early Stopping & LR Scheduler

The training loop uses:
	•	ReduceLROnPlateau: reduces learning rate if validation accuracy plateaus
	•	Custom Early Stopping logic to stop training if no improvement is seen after N epochs

These techniques help combat overfitting, especially on small datasets like this.

⸻

📈 Evaluation Metrics

	•	Accuracy
	•	Binary Cross Entropy Loss
	•	Confusion Matrix
	•	ROC-AUC Score (optional)
	•	Per-class precision, recall, F1 (via classification_report)

⸻

📊 Example Visualization

Each notebook generates:
	•	Training vs Validation Loss / Accuracy curves
	•	Prediction Samples with confidence scores
	•	Class Distribution Plots

⸻

💡 Tips

	•	Run on GPU for best performance (e.g., Google Colab or local CUDA setup)
	•	Ensure all transformations match model input expectations (e.g., 224x224 for VGG16)
	•	You can fine-tune later layers for potentially better accuracy

⸻

🙋‍♂️ Author

Brian Lim Yong Jenq, Khairul Shabir, Tan Zi Hui, Turrag Dewan – SUTD – Applied Deep Learning, Term 8

Project guidance: Prof. Priti 
