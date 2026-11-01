# RSNA Knee MRI Abnormality Detection

A deep learning project where medical imaging meets real-world constraints: predicting 12 knee abnormalities from MRI scans for the RSNA Kaggle competition.

## About

This project builds an end-to-end machine learning pipeline that reads raw knee MRI scans (DICOM format) and predicts the probability of 12 conditions, including ACL tears, meniscus tears, osteoarthritis, joint effusion, and fractures. It reflects my approach to learning through hands-on experience: starting with a simple working baseline, then improving it step by step, and solving the real data and platform problems that come up along the way.

This is my second AI/ML project, building on the fundamentals from my first one (a hybrid Minimax + Random Forest tic-tac-toe predictor).

## Results

| Version | Approach | Public Score (Macro-AUC) |
|---|---|---|
| Level 1 Baseline | 1 slice per study, single train/validation split | 0.543 |
| **Level 2 Final** | **3 slices, augmentation, 5-fold cross-validation, ensembling** | **0.603** |

Score is the competition's official metric: macro-averaged AUC across all 12 conditions (0.5 = random guessing, 1.0 = perfect).

## The Challenge

- Only **58 of 4,407** training studies (1.3%) came with ground-truth labels.
- The rest only had free-text radiology reports, written in Spanish, Dutch, French, Turkish, Bulgarian, Greek, Croatian, German, and English.
- The competition is a Code Competition: internet access is disabled during scoring, which affects how pretrained model weights can be used.

## Objectives

- Build a complete, working medical image classification pipeline from raw DICOM files to a valid submission.
- Apply core ML techniques correctly: transfer learning concepts, data augmentation, cross-validation, and ensembling.
- Handle messy real-world data: scarce labels, multilingual text, and inconsistent file structure.
- Learn how to build reproducible notebooks that run cleanly from top to bottom.

## What I Built

- **Multilingual weak-labeling pipeline:** language detection, local translation with NLLB-200, and keyword-based label guessing, validated against the 58 gold-labeled studies.
- **DICOM image pipeline:** slice ordering by `InstanceNumber` metadata, automatic selection of the sagittal fluid-sensitive series, resizing and normalization.
- **Multi-label CNN:** a ResNet18 adapted for multi-slice MRI input and 12 simultaneous predictions.
- **Training setup:** augmentation (flip, rotation, brightness/contrast), 5-fold cross-validation, and ensembling across all fold models.
- **Competition-ready output:** a metric matching Kaggle's grading and a correctly formatted `submission.csv`.

## Tech & Tools

- **Languages & Libraries:** Python, PyTorch, torchvision, pydicom, OpenCV, scikit-learn, albumentations, pandas, NumPy
- **NLP:** Hugging Face Transformers (NLLB-200), langdetect
- **Compute:** Kaggle Notebooks (GPU T4 x2)
- **Version Control:** Git and GitHub

## Key Lessons

- Always inspect real data before trusting assumptions about it.
- With very small datasets, one train/validation split can be misleading; cross-validation gives a trustworthy estimate.
- A notebook that works interactively is not the same as one that runs cleanly from scratch.
- Sometimes a simpler, fully compliant solution beats a more sophisticated one that conflicts with platform rules.

