# Computer Vision - Portfolio Exam 2

# Assignment 2: Edge AI for Industry 4.0 - Optimizing Models for On-Device Inference

**Submitted by:**

* Riya Biju - 10000742
* Harsha Sathish - 10001000
* Harshith Babu Prakash Babu - 10001191

---

## Project Overview

This project implements Edge AI optimization techniques for deploying Personal Protective Equipment (PPE) detection models on resource-constrained edge devices in Industry 4.0 environments. The system demonstrates various model compression strategies, specifically pruning techniques, to achieve efficient on-device inference while maintaining acceptable detection accuracy.

**Key Features:**
* Lightweight CNN architecture (MobileNet) for PPE detection
* Multiple pruning strategies: Magnitude-based, Iterative, and One-shot pruning
* Comprehensive comparison of model size, inference time, and accuracy trade-offs
* Edge AI system architecture design for industrial deployment
* Analysis of optimization strategies for latency, energy, and accuracy

---

## Prerequisites

**Dataset:**

Download the PPE Detection Dataset from Mendeley Data:
* Dataset URL: https://data.mendeley.com/datasets/zkzghjvpn2/6
* Extract the dataset to your working directory
* The dataset contains images of workers with/without proper PPE for industrial safety monitoring

**Python Environment:**

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## Repository Structure

```
CV_Portfolio2_Riya_Harsha_Harshith/
│
├── Assignment-1_PneumoniaDetection/           # Assignment 1 folder
│
└── Assignment-2_EdgeAI/                        # Assignment 2 folder
    ├── README.md                               # This file
    ├── Portfolio2_Part2.pdf                    # PDF Documentation
    ├── Portfolio2_task2.ipynb                  # Main Jupyter notebook
    ├── requirements.txt                        # Python dependencies
    │
    ├── baseline_model.pth                      # Original MobileNet model (no pruning)
    ├── best_pruned_model.pth                   # Best performing pruned model
    ├── model_magnitude_pruned.pth              # Magnitude-based pruning result
    ├── model_iterative_pruned.pth              # Iterative pruning result
    ├── model_oneshot_pruned.pth                # One-shot pruning result
    │
    ├── pruning_results.png                     # Comparison of pruning methods
    ├── pruning_comprehensive_results.png       # Detailed analysis visualizations
    └── Presentation.pdf                        # Project presentation slides
```

---

## Results Overview

### Output Files:

1. **baseline_model.pth**
   * Original MobileNet without any compression
   * Serves as reference for accuracy and size comparisons

2. **model_magnitude_pruned.pth**
   * Result of magnitude-based pruning heuristic
   * Shows accuracy-size trade-off for simple pruning

3. **model_iterative_pruned.pth**
   * Result of gradual, iterative pruning approach
   * Typically achieves better accuracy retention

4. **model_oneshot_pruned.pth**
   * Result of single-step aggressive pruning
   * Fastest compression but may sacrifice accuracy

5. **best_pruned_model.pth**
   * Model with optimal balance of size and accuracy
   * Selected based on evaluation metrics

6. **pruning_results.png**
   * Comparative bar charts showing:
     - Model size reduction across methods
     - Accuracy comparison at different sparsity levels
     - Inference time improvements

7. **pruning_comprehensive_results.png**
   * Detailed analysis including:
     - Accuracy vs. sparsity curves
     - Compression ratio vs. accuracy trade-offs
     - Layer-wise pruning sensitivity analysis

---

## Dependencies

Key libraries required (see `requirements.txt` for complete list):
* PyTorch (deep learning framework)
* torchvision (MobileNet and transforms)
* torch.nn.utils.prune (pruning utilities)
* numpy (numerical operations)
* matplotlib (visualization)
* opencv-python (image processing)
* Pillow (image loading)
* scikit-learn (evaluation metrics)
* tqdm (progress bars)

---

## Acknowledgments

* **Instructor:** Prof. Dr. Dominik Seuß - Computer Vision Course, THWS
* **Dataset:** PPE Detection Dataset from Mendeley Data
* **Framework:** PyTorch model compression utilities

