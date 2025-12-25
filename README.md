# SkinCancerDetection

# Project Overview 
This project implements a high-performance Deep Learning pipeline to classify skin lesions into two categories: Cancerous and Non-Cancerous. Utilizing the HAM10000 dataset, the system leverages Transfer Learning with the DenseNet201 architecture to achieve high diagnostic reliability. The primary goal of this research-oriented project is to assist in the early detection of skin cancer, specifically addressing common challenges in medical AI such as class imbalance and vanishing gradients. 

# Key Features 
● Architecture: Implemented DenseNet201 to maximize feature reuse and strengthen gradient flow. 
● Data Balancing: Integrated Synthetic Upsampling to solve the inherent class imbalance in medical datasets. 
● Transfer Learning: Utilized pre-trained ImageNet weights for robust feature extraction followed by domain-specific fine-tuning. 
● Evaluation Metrics: Comprehensive analysis using Confusion Matrices, F1-Score, and Recall to prioritize patient safety (minimizing False Negatives).

# System Architecture 
The model follows a structured end-to-end pipeline: 
1. Data Ingestion: Mapping ISIC metadata to dermatoscopic images. 
2. Preprocessing: Image resizing (128 X 128), normalization (0.0-1.0), and data splitting. 
3. Model Build: 
   ○ Base: DenseNet201 (Frozen layers). 
   ○ Head: Global Average Pooling -> Dropout (0.15) -> Dense Layers (128, ReLU) -> Batch Normalization -> Softmax Output. 
4. Training: Optimized using the Adam Optimizer and Sparse Categorical Cross-Entropy. 

# Dataset Details
● Name: HAM10000 ("Human Against Machine with 10015 training images") 
● Source: ViDIR Group, Medical University of Vienna (2018). 
● Classes: Originally 7 classes, mapped to Binary Classification (Cancer vs. Not Cancer). 
● Size: 10,015 high-resolution dermatoscopic images.

# Tech Stack 
● Language: Python 3.x 
● Deep Learning: TensorFlow, Keras 
● Data Processing: Pandas, NumPy, Scikit-learn 
● Computer Vision: OpenCV (cv2), PIL 
● Visualization: Matplotlib, Seaborn 

# Installation & Usage 
1. Clone the Repository: 
Bash 
git clone https://github.com/Srivalli-M12/SkinCancerDetection.git 
cd SkinCancerDetection 
2. Install Dependencies: 
Bash 
pip install tensorflow pandas numpy opencv-python matplotlib seaborn scikit-learn 
3. Run the Notebook: 
Open the .ipynb file in Google Colab or Jupyter Notebook. Ensure your dataset is mounted on Google Drive as per the path configurations in the code.

# Results 
The model demonstrates excellent performance in distinguishing malignant cases. By utilizing DenseNet's dense connections, the model maintains high Recall, ensuring that cancerous lesions are rarely missed. 


