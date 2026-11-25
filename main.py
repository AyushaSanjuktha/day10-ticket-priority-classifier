
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline

SEED = 42
LABELS = ["Low", "Medium", "High", "Critical"]

path = sys.argv[1] if len(sys.argv) > 1 else "tickets_labelled.csv"
df = pd.read_csv(path)
df["text"] = df["title"] + ". " + df["description"]
X, y = df["text"], df["priority"]

# 1) Stratified split. The vectoriser lives INSIDE the pipeline, so it only
#    ever learns its vocabulary from training data (no leakage).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=SEED)),
])

# 2) Cross-validation on the training set only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
print(f"5-fold CV macro-F1: {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

# 3) Fit and evaluate once on the untouched test set
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
report = classification_report(y_test, pred, labels=LABELS, digits=3)
cm = pd.DataFrame(confusion_matrix(y_test, pred, labels=LABELS),
                  index=[f"true_{l}" for l in LABELS],
                  columns=[f"pred_{l}" for l in LABELS])
test_f1 = f1_score(y_test, pred, average="macro")
print("\nTest set report:\n", report)
print("Confusion matrix:\n", cm)

# 4) Error analysis: out-of-fold predictions on the training set give enough
#    mistakes to inspect (the test set alone has very few).
oof = cross_val_predict(pipe, X_train, y_train, cv=cv)
errors = pd.DataFrame({"text": X_train, "true": y_train, "predicted": oof})
errors = errors[errors["true"] != errors["predicted"]]
sample = errors.sample(min(20, len(errors)), random_state=SEED)
sample.to_csv("misclassified_20.csv", index=False)
print(f"\nOut-of-fold errors in train: {len(errors)} of {len(X_train)}")
print(errors.groupby(["true", "predicted"]).size()
      .sort_values(ascending=False).head(8))

# 5) Save model + metrics
joblib.dump(pipe, "ticket_priority_model.joblib")
with open("metrics.txt", "w") as f:
    f.write(f"CV macro-F1: {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}\n")
    f.write(f"Test macro-F1: {test_f1:.3f}\n\n{report}\n{cm}\n")
