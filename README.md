ML Model Evaluation Assignment
Project objective
Compare a baseline Logistic Regression classifier with a Random Forest classifier
on the roommate compatibility dataset. The experiment investigates whether any
observed accuracy improvement is larger than would reasonably be expected from
test-sample variation.
Dataset
Place training_data.csv inside data/.
The target is created from compatibility_score:
0: score below 70 (Not Compatible)
1: score 70 or above (Compatible)
The threshold is an experimental definition and should be stated in the report.
IDs, names, and free-text bios are excluded from the initial model features.
Duplicate unordered student pairs are kept together using a grouped split.
Setup (macOS / VS Code)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python assignment.py
​
In VS Code, select the interpreter at .venv/bin/python if prompted.
Outputs
The script creates these files inside outputs/:
model_metrics.csv
test_predictions.csv
statistical_test_results.csv
baseline_confusion_matrix.png
improved_confusion_matrix.png
bootstrap_difference_distribution.png
Statistical methods
StratifiedGroupKFold: approximately 80/20 split, with duplicate student pairs
kept in one partition.
Paired bootstrap: 5,000 resamples of the held-out test rows.
95% percentile confidence interval for accuracy(Random Forest) minus
accuracy(Logistic Regression).
One-sided exact McNemar test on paired correctness outcomes, at alpha = 0.05.
Important interpretation
Random Forest is a candidate improved model, not a guaranteed improvement.
Use the measured test metrics, confidence interval, and p-value to make the
final decision. The bootstrap resamples test rows; because some students can
appear in more than one distinct pair, this interval does not fully account
for dependence between different pairs sharing a student. Mention this as a
limitation in the report.
Reproducibility
The script uses fixed random seeds. Keep the dataset unchanged and use the same
Python package versions when reproducing results.