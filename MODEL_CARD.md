# Model Card - Ticket Priority Classifier (Classic ML Baseline)

**Author:** Ayusha Sanjuktha Chekka | **Day 10, AI02** | **Seed:** 42

## Purpose
A cheap, fast baseline that predicts ticket priority (Low / Medium / High / Critical)
from the ticket text. It is a reference to compare against LLM classification later.

## Model
TF-IDF (1-2 grams, min_df=2, sublinear tf) + Logistic Regression
(class_weight="balanced"), built as one scikit-learn Pipeline so the vectoriser is
fitted only on training data (no leakage).

## Data
- File: `tickets_labelled.csv`, 800 synthetic tickets given by the instructor.
- Input text: title + description. Label: priority.
- Class counts: Medium 304, Low 244, High 183, Critical 69 (imbalanced).
- Only 20 distinct titles; descriptions differ mainly by a short random ending
  (for example "Urgent!!", "Please help.", "Thanks.").
- Split: 80% train (640) / 20% test (160), stratified. 5-fold stratified CV on train.

## Metrics
| Measure | Value |
| --- | --- |
| 5-fold CV macro-F1 (train) | 0.940 +/- 0.009 |
| Test accuracy | 0.894 |
| Test macro-F1 | 0.888 |

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Low | 0.923 | 0.980 | 0.950 | 49 |
| Medium | 0.885 | 0.885 | 0.885 | 61 |
| High | 0.853 | 0.806 | 0.829 | 36 |
| Critical | 0.923 | 0.857 | 0.889 | 14 |

Macro-F1 is used as the main metric because the classes are imbalanced.

## Error analysis (20 misclassified, out-of-fold)
- Most errors are between neighbouring levels (High/Medium, Low/Medium, Critical/High).
- Many wrong tickets have the same text as correctly predicted ones but a different
  label (for example "Invoice amounts wrong" is High 32 times, Medium 6 times).
  The model picks the majority label, so these look like label noise, not model faults.
- The random endings ("Urgent!!" and similar) do not predict priority in this data.

## Limits
- Synthetic data with very few distinct titles. The test set shares titles with the
  training set, so scores are probably higher than on real, new tickets.
- Test set is small (14 Critical tickets), so per-class scores can move a lot.
- Cannot use meaning or context beyond words seen in training.
- Should not be used for real triage without testing on real data.

## Files
`ticket_priority_classifier.py`, `ticket_priority_model.joblib`, `metrics.txt`,
`misclassified_20.csv`, `model_card.md`