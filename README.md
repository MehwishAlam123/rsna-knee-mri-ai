RSNA Knee MRI Abnormality Detection

A multimodal machine learning project built for the RSNA Knee MRI Kaggle competition — predicting 12 knee abnormalities (ACL tears, meniscus tears, osteoarthritis, effusion, fractures, and more) from MRI scans and their accompanying radiology reports.

This is my second AI/ML project, building on the fundamentals from a hybrid Minimax + Random Forest tic-tac-toe predictor (my first project).

Final Result
Version	Approach	Public Leaderboard Score
Baseline (Level 1)	1 slice/study, single train/val split	0.543
Final (Level 2)	3 slices/study, augmentation, 5-fold CV, ensembling	0.603

(AUC ranges 0–1, where 0.5 = random guessing and 1.0 = perfect. Score = 12-condition macro-averaged AUC, matching the competition's official metric.)

The Challenge

The competition provides ~4,400 knee MRI studies, but only 58 of them (1.3%) came with pre-filled ground-truth labels. The rest only had free-text radiology reports — written by radiologists across multiple countries in Spanish, Dutch, French, Turkish, Bulgarian, Greek, Croatian, and German.

This meant the real first problem wasn't computer vision — it was figuring out how to get usable training labels at all.

What I Built
1. A multilingual weak-labeling pipeline
Detected each report's language automatically (langdetect)
Translated non-English reports to English using a local NLLB-200 translation model (no external API, avoids rate limits)
Built regex-based keyword labelers for all 12 target conditions, validated against the 58 gold-labeled studies
Result: about 5 of 12 labels showed meaningful signal over an "always guess negative" baseline; validated but not yet integrated into the final training set — a clear next step for future work
2. An image-based CNN pipeline
dicom_utils.py — reads raw DICOM MRI slices, correctly sorted by InstanceNumber (not filename, since filenames are random UIDs)
select_series.py — automatically picks the sagittal, fluid-sensitive series per study (the standard view for most of these findings), with a fallback rule
dataset.py — a PyTorch Dataset that pairs each study's MRI slices with its 12 labels, with optional augmentation (flips, rotation, brightness/contrast)
model.py — a ResNet18 CNN, adapted for multi-slice MRI input and 12-label multi-output prediction
metric.py — a macro-AUC scorer matching Kaggle's exact grading formula
train.py — 5-fold grouped cross-validation training loop, with per-fold checkpointing
predict.py — generates a competition-ready submission.csv by ensembling predictions across all 5 fold models
Key Engineering Challenges Solved
Label scarcity: discovered only 58/4407 studies had usable labels; built a translation + weak-labeling system to address it
DICOM slice ordering: filenames were random UIDs, not sequential — fixed by sorting on the InstanceNumber DICOM metadata field instead
Validation instability: a single train/val split gave wildly different scores run-to-run (0.53 vs 0.59) on identical code with only 58 samples — resolved with 5-fold cross-validation and ensembling for a stable, trustworthy estimate
No-internet Code Competition constraint: this competition disables internet access during scoring; after working through several approaches (a locally cached weights file didn't persist into fresh commit environments), the final model trains from scratch (no pretrained weights) to guarantee compliance while still beating the pretrained-but-unstable earlier baseline
Tech Stack
PyTorch / torchvision — CNN model and training
pydicom — medical image (DICOM) reading
facebook/nllb-200-distilled-600M (Hugging Face Transformers) — multilingual report translation
langdetect — language detection
scikit-learn — AUC scoring, K-Fold splitting
albumentations — image augmentation
pandas / numpy / OpenCV — data handling and image processing
Trained on Kaggle Notebooks (GPU T4 x2)
Project Structure
├── dicom_utils.py       # DICOM reading + slice sorting
├── select_series.py     # picks the right MRI series per study
├── dataset.py           # PyTorch Dataset class with augmentation support
├── model.py             # ResNet18-based multi-label CNN
├── metric.py            # macro-AUC scorer (matches Kaggle's grading)
├── train.py             # 5-fold cross-validation training loop
├── predict.py           # ensembled prediction + submission.csv generation
└── weak_labeling/       # translation + regex-based report labeling (exploratory)
What I Learned
Real-world datasets are often far messier than a first read suggests — assuming "most rows are labeled" turned out to be wrong by two orders of magnitude
Weak/noisy labeling from unstructured text is a legitimate, hard sub-problem on its own, not just a preprocessing afterthought
With very small datasets, a single train/validation split can be dangerously misleading — cross-validation isn't optional polish, it's necessary for a trustworthy result
Real competition platforms come with real constraints (no-internet Code Competition rules) that can conflict with standard ML practices like downloading pretrained weights — sometimes the right engineering call is a simpler, fully-compliant approach over a more sophisticated one that doesn't reliably run
DICOM metadata (like InstanceNumber) matters — naive assumptions (like sorting by filename) silently produce wrong results
Getting a notebook to work interactively is not the same as getting it to work in a clean, reproducible "run all" commit — leftover code from earlier experiments is a real, recurring failure mode worth actively guarding against
