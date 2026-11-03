<div align="center">
  <h1>Roommate Compatibility Prediction</h1>
  <h2> ML Model Evaluation Assignment</h2>
  <p>Project objective:
Compare a baseline Logistic Regression classifier with a Random Forest classifier
on the roommate compatibility dataset. The experiment investigates whether any
observed accuracy improvement is larger than would reasonably be expected from
test-sample variation.</p>
</div>
<p>
<b>Setup (macOS / VS Code):</b>
<ul>python3 -m venv .venv</ul>
<ul>source .venv/bin/activate</ul>
<ul>python -m pip install --upgrade pip</ul>
<ul>pip install -r requirements.txt</ul>
<ul>python assignment.py</ul>
​<ul>In VS Code, select the interpreter at .venv/bin/python if prompted.</ul></p>

<p><ul>The target is created from compatibility_score:<br>0: score below 70 (Not Compatible)<br>1: score 70 or above (Compatible)
<br>The threshold is an experimental definition and should be stated in the report.
IDs, names, and free-text bios are excluded from the initial model features.
Duplicate unordered student pairs are kept together using a grouped split.</ul></p>


<p><b>Outputs</b>
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
<ul>One-sided exact McNemar test on paired correctness outcomes, at alpha = 0.05.</ul>
<b>Important interpretation</b><br>
<ul>Random Forest is a candidate improved model, not a guaranteed improvement.
Use the measured test metrics, confidence interval, and p-value to make the
final decision. The bootstrap resamples test rows; because some students can
appear in more than one distinct pair, this interval does not fully account
for dependence between different pairs sharing a student. Mention this as a
limitation in the report.</ul>
<b>Reproducibility</b>
<ul>The script uses fixed random seeds. Keep the dataset unchanged and use the same
Python package versions when reproducing results.</ul></p>
