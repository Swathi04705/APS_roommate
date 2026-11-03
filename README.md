<div align="center">
  <h1>Roommate Compatibility Prediction</h1>
  <h2> ML Model Evaluation Assignment</h2>
  <p>Project objective:
Compare a baseline Logistic Regression classifier with a Random Forest classifier
on the roommate compatibility dataset. The experiment investigates whether any
observed accuracy improvement is larger than would reasonably be expected from
test-sample variation.</p>
</div>
<details>
Setup (macOS / VS Code)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python assignment.py
​In VS Code, select the interpreter at .venv/bin/python if prompted.

The target is created from compatibility_score:<br>0: score below 70 (Not Compatible)<br>1: score 70 or above (Compatible)
<br>The threshold is an experimental definition and should be stated in the report.
IDs, names, and free-text bios are excluded from the initial model features.
Duplicate unordered student pairs are kept together using a grouped split.
</details>

<details> <b>Outputs</b><br>
The script creates these files inside outputs/:
<ul>model_metrics.csv</ul>
<ul>test_predictions.csv</ul>
<ul>statistical_test_results.csv</ul>
<ul>baseline_confusion_matrix.png</ul>
<ul>improved_confusion_matrix.png</ul>
<ul>bootstrap_difference_distribution.png</ul>
<b>Statistical methods</b>
<ul>StratifiedGroupKFold: approximately 80/20 split, with duplicate student pairs
kept in one partition.</ul>
<ul>Paired bootstrap: 5,000 resamples of the held-out test rows.
95% percentile confidence interval for accuracy(Random Forest) minus
accuracy(Logistic Regression).</ul>
One-sided exact McNemar test on paired correctness outcomes, at alpha = 0.05.<br>
<b>Important interpretation</b><br>
Random Forest is a candidate improved model, not a guaranteed improvement.
Use the measured test metrics, confidence interval, and p-value to make the
final decision. The bootstrap resamples test rows; because some students can
appear in more than one distinct pair, this interval does not fully account
for dependence between different pairs sharing a student. Mention this as a
limitation in the report.<br>
<b>Reproducibility</b><br>
The script uses fixed random seeds. Keep the dataset unchanged and use the same
Python package versions when reproducing results.</details>
