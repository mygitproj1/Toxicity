🛡️ ToxicShield AI: Enterprise Toxicity Detection Pipeline
Built as a foundational component for Enterprise AI Governance systems, this project provides a cost-effective, locally hosted MLOps pipeline for evaluating text data. It performs multi-label classification to identify and categorize toxicity in chat logs (Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate) without relying on external APIs, ensuring complete data privacy.

🎯 Project Overview
This repository demonstrates the evolution from a baseline statistical model to a state-of-the-art Deep Learning architecture, fully optimized to run locally on an Intel Arc GPU.

Phase 1: Baseline - TF-IDF Vectorization with Logistic Regression.

Phase 2: Deep Learning - Fine-tuning roberta-base via Hugging Face Transformers.

Phase 3: Custom Architecture - (In Progress) A ground-up PyTorch Transformer model for layer-by-layer architectural comparison.

🛠️ Tech Stack & MLOps Infrastructure
Deep Learning Framework: PyTorch with Intel Extension (ipex)

NLP Models: Hugging Face Transformers (roberta-base)

Experiment Tracking: MLflow

User Interface: Streamlit

Hardware Acceleration: Intel Arc GPU (XPU)

⚙️ Hardware & Installation
This project is explicitly configured to utilize Intel Arc GPUs (NPUs/XPUs) on Windows to bypass the need for expensive cloud compute.

Prerequisites:

Python 3.10+

VS Code (recommended)

An active Python Virtual Environment (venv)

Step 1: Install Core Dependencies

Bash
pip install pandas numpy scikit-learn transformers[torch] datasets evaluate mlflow streamlit
Step 2: Configure Intel PyTorch (XPU)
Due to specific versioning requirements between PyTorch and Intel's C++ runtimes, install the extension using these exact commands:

Bash
# 1. Install PyTorch 2.6.0 explicitly
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/xpu

# 2. Force install the Intel Extension (bypassing pip dependency conflicts)
pip install intel-extension-for-pytorch==2.6.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ --no-deps

# 3. Patch missing/deprecated dependencies required by the Intel extension
pip install setuptools==69.5.1
pip install triton==3.2.0
🚀 Usage
1. Model Training & Tracking
To train the RoBERTa model on your local Intel GPU and automatically log metrics (Macro F1-Score, Loss, Hyperparameters):

Bash
python toxicity_roberta_xpu.py
2. View MLflow Dashboard
To compare model runs and view training curves:

Bash
mlflow ui
Navigate to http://localhost:5000 in your web browser.

3. Launch the Streamlit Sandbox
To test the locally saved model with real-time inference:

Bash
streamlit run app.py
🔧 Troubleshooting Guide
If you encounter environment issues while setting up the Intel Arc GPU, reference these known fixes:

ModuleNotFoundError: No module named 'pkg_resources': The Intel extension relies on an older version of setuptools. Fix: pip install setuptools==69.5.1.

ImportError: cannot import name 'AttrsDescriptor' from 'triton.compiler.compiler': PyTorch 2.6 is incompatible with Triton 3.3.0. Fix: pip install triton==3.2.0.

pip's dependency resolver does not currently take into account...: Intel SYCL/C++ runtime mismatch. Fix: Use the --no-deps flag when installing intel-extension-for-pytorch to bind it to PyTorch's native downloads.

Terminal Freezes / Unresponsive to Ctrl + C: When falling back to CPU training, the process may lock the terminal. Use Windows Task Manager to end the python.exe background process.

🗺️ Future Roadmap
[ ] Implement a from-scratch PyTorch Transformer architecture for comparative analysis.

[ ] Integrate DVC (Data Version Control) to track datasets alongside code.

[ ] Set up GitHub Actions for automated CI/CD pipeline testing.