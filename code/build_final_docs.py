"""Generate concise entry points from the recorded experiment and reviewed findings."""
from pathlib import Path
import json
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/final'


def main():
    summary = json.loads((OUT / 'analysis_summary.json').read_text())
    findings = json.loads((OUT / 'reviewed_findings.json').read_text())
    models = {row['family']: row for row in summary['models']}
    table = ['| Model | Outer CV F1 ± SD | Test F1 | Test precision | Test recall |',
             '|---|---:|---:|---:|---:|']
    for key in ['original_metadata', 'engineered_metadata', 'tfidf_text', 'tfidf_combined',
                'minilm_text', 'minilm_combined', 'target_encoded_metadata',
                'target_minilm_combined', 'random_forest', 'gradient_boosting', 'mlp',
                'endterm_rf_reference', 'svm_reference', 'dummy_majority']:
        row = models[key]
        precision = Decimal(str(row['test_precision'])).quantize(Decimal('.001'), rounding=ROUND_HALF_UP)
        table.append(f'| {row["name"]} | {row["cv_f1_mean"]:.3f} ± {row["cv_f1_sd"]:.3f} | '
                     f'{row["test_f1"]:.3f} | {precision} | {row["test_recall"]:.3f} |')
    table = '\n'.join(table)
    cases = '\n\n'.join('**'+case['name']+'** — '+case['text'] for case in findings['cases'])
    limits = '\n\n'.join(findings['limitations'])
    public = f'''# TVmaze: can a synopsis improve rating classification?

**[Bekzat Rashiduly](https://github.com/gojosatorukz) — an individual ML experiment developed from an Astana IT University course project.** I own the technical development: data collection, hypotheses, implementation, evaluation and the local prediction app. The course presentation is delivered with classmates.

The question is specific: **does a series synopsis improve prediction of a TVmaze rating ≥8/10 when structured metadata are already available?** I compare metadata-only, text-only and combined pipelines on the same saved objects and splits, including a simple learned TF-IDF baseline and a frozen pretrained MiniLM encoder.

{findings['conclusion']}

![Representation comparison](figures/representation_comparison.png)

## Data and prediction target

The immutable 6 October 2026 TVmaze snapshot contains 14,055 raw series. Filtering leaves **2,339 rated English scripted series**, including **404 positives (17.3%)**. One row represents one series. The 1,732 otherwise eligible series with unknown ratings are excluded. All retained series have a synopsis linked to the same show ID.

The target is `rating >= 8`. Model inputs are premiere year, runtime, network, web channel, broadcaster/platform country, distribution, genres and, in the text experiments, the synopsis. Rating, labels, titles, IDs, URLs, status and popularity are excluded from prediction inputs. IDs remain available for audit and matching.

The saved split is **1,871 training / 468 test**, with five saved stratified training folds. Raw responses, collection timestamps and hashes are preserved. The test set was reported in earlier course stages: it is a **reused benchmark**, not an unseen prospective test.

## Experimental design

Positive-class F1 is primary. The majority classifier achieves about 83% accuracy while recovering none of the highly rated series. Precision, recall, ROC-AUC, average precision and confusion matrices provide complementary views.

Five outer folds evaluate the tuned procedure. Each outer fit contains an independent three-fold inner search. A separate five-fold search on the complete training partition chooses deployment settings. All learned preprocessing fits within its corresponding training partition. The training-CV choice is recorded in `selection_before_test.json` before this Final test evaluation. A bootstrap interval resamples the same test objects in pairs; it conditions on the fitted models and does not measure retraining uncertainty.

The metadata study adds genre count, genre combinations, premiere decade and runtime buckets. A separate cross-fitted target encoder tests sparse channel categories without using a training row’s own label to encode that row. Random Forest and Gradient Boosting each use **256 exhaustive configurations**; an MLP uses **64**, spanning four architectures, four L2 penalties and four learning rates. Complete Cartesian grid tables and selected boundary positions accompany the results.

For text, TF-IDF learns word unigrams and bigrams within each fitting fold. An LSA control reduces this representation to 48 dimensions. Frozen **all-MiniLM-L6-v2** generates 384-dimensional vectors using masked mean pooling and L2 normalisation. A logistic classifier learns the rating task. The pinned, quantised ONNX encoder runs locally; its six transformer layers remain frozen. It reads up to 256 tokens, truncating 51 training and 12 test descriptions.

## Results

{table}

{chr(10).join(findings['results'])}

The full comparison includes 17 saved models. The app chooses its metadata, text and combined variants by outer training CV. High test scores are not used to choose the primary model.

![Learning curves](figures/mlp_learning.png)

The MLP restores epoch **{summary['mlp']['best_epoch']}**, the absolute validation-loss minimum. Training loss continues to improve after that point while validation loss worsens. The fitting–validation loss gap increases from **{summary['mlp']['gap_at_best']:.3f}** to **{summary['mlp']['gap_at_stop']:.3f}** by epoch {summary['mlp']['stopped_epoch']}. The saved model retains the selected checkpoint and the preprocessing fitted on its fitting subset. The earlier epoch-selection rule was corrected to retain even a small new absolute minimum.

## What the mistakes show

{cases}

{findings['additional_error_note']}

Feature replacements and sentence occlusion measure sensitivity of a fitted model. They do not establish causes of audience ratings. Exact inputs, training-group counts and score changes are saved in `error_evidence.json`; per-series predictions make the examples auditable.

## Run from a fresh clone

Python **3.12** is tested. Create an isolated environment and extract both included archives:

```bash
git clone https://github.com/gojosatorukz/tvmaze-series-classification.git
cd tvmaze-series-classification
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-final.txt
python -m zipfile -e data_snapshot.zip .
python -m zipfile -e final_artifacts.zip .
python app.py
```

Open <http://127.0.0.1:8765>. On Windows, use `.venv\\Scripts\\activate`. The app loads saved pipelines and accepts a new series. Missing text falls back to metadata; missing numeric values use training medians; invalid values receive an input error. Model scores are uncalibrated. A fictional input has no known correct rating.

The archives contain the data and trained classifiers. MiniLM weights, tokenizer, pinned revision and license are in `assets/minilm/`. After dependency installation, prediction works offline. Only load the trusted model files supplied with this project.

![Local saved-model app](figures/app_demo.jpg)

```bash
# Verify snapshot hashes, complete grids, all saved predictions and the MLP checkpoint
python code/run_final.py --verify-only
python -m unittest discover -s tests -v

# Execute the explanatory notebook and export HTML
python code/run_final_notebook.py

# Expensive: reproduce all exhaustive searches and analysis
python code/run_final.py --fresh --workers 4
```

The last command creates a new reproduction directory under `.runtime/`, preserves the included results and retrains from scratch. It takes hours, rather than the seconds needed for saved-model inference. To resume, run without `--fresh` from the printed reproduction directory. Without `--fresh`, matching completed searches resume instead of retraining. Seeds and numerical-library threads are fixed. The included search tables document every evaluated candidate. Hardware and library changes may cause small floating-point variation.

## Main files

- `Final_TVSeries.ipynb`: executed walkthrough, actual results, error evidence and inference.
- `app.py`, `templates/`, `static/`: local saved-model application.
- `code/project.py`, `code/collect_data.py`: original data collection, cleaning and split.
- `code/final_models.py`: engineered metadata, complete teacher grids and restored MLP checkpoint.
- `code/final_text.py`: frozen local MiniLM representations and combined pipelines.
- `code/target_features.py`: cross-fitted category encoding experiment.
- `code/final_experiments.py`: resumable nested exhaustive searches.
- `code/audit_final_data.py`: reproducible text coverage, matching and truncation checks.
- `code/final_analysis.py`: training-only choices, test evaluation, paired intervals and case diagnostics.
- `code/verify_final.py`, `tests/`: integrity, saved-model reproduction and app behaviour checks.
- `data_snapshot.zip`, `final_artifacts.zip`: immutable data and recorded trained results.
- `Endterm_TVSeries.ipynb`: historical metadata-only stage; the Final notebook is the current entry point.

## Limits and next experiment

{limits}

A new temporal or franchise-grouped evaluation would test generalisation more convincingly. Threshold selection, if added, must happen inside training folds. The current evidence does not validate predictions for future or unrated shows.

## Sources and licenses

Data: [TVmaze API](https://www.tvmaze.com/api), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Cleaning strips HTML, selects eligible objects and adds the binary label. Attribution and ShareAlike apply to redistributed data; see `DATA_LICENSE.txt`.

Pretrained encoder: [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0. The exact revision and SHA-256 hashes are in its manifest. The data license does not grant a separate software license.

Method references: [TF-IDF feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html), [Sentence-BERT](https://arxiv.org/abs/1908.10084), [MiniLM](https://arxiv.org/abs/2002.10957), [nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html), [Target Encoder cross fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html).
'''
    (ROOT / 'README_FINAL.md').write_text(public, encoding='utf-8')
    feedback = '# Final changes and evidence\n\n'
    for item in findings['feedback']:
        feedback += '## '+item['issue']+'\n\n'+item['response']+'\n\n'
    feedback += ('The complete Endterm comment is in `TEACHER_FEEDBACK.txt`. The LMS Midterm view '
                 'did not expose the complete original comment text. This change log responds to the '
                 'Midterm issues identified in the Endterm feedback.\n')
    (ROOT / 'docs/CHANGELOG_FINAL.md').write_text(feedback, encoding='utf-8')
    criteria = '''# Final rubric: evidence map

| Criterion | Evidence |
|---|---|
| New lawful text and learned model, 20 | Original matched synopses, TVmaze attribution, pinned MiniLM + learned logistic classifier, TF-IDF comparator; report sections 2–4 |
| Comparison and error analysis, 15 | Same IDs/split/folds; metadata/text/combined metrics; paired intervals; `error_evidence.json`; report sections 5–7 |
| Local app, 15 | `app.py`, saved models and preprocessing; new-input form; missing/invalid input handling; `tests/test_app.py` |
| Written report, 15 | `report/Final_Report.pdf`: question, existing approaches, methods, results, limitations, actual contributions and required AI-tool disclosure |
| Reproducibility, 10 | Complete code/data/models/weights; pinned Python dependencies; `run_final.py`; `verify_final.py`; executed notebook and HTML |
| Presentation, 10 | Nine slides, seven-minute shared talk, about 2.5-minute local app demo; English speaker text |
| Individual understanding, 15 each | `LEARNING_GUIDE_RU.md`, actual model launch and oral questions; the grade depends on each student’s answers |

The teacher’s additional full-grid, architecture, checkpoint and error-explanation requests are mapped in `CHANGELOG_FINAL.md`. Source documents were read as data; see `FINAL_INPUT_REVIEW.txt` for inspection coverage.
'''
    (ROOT / 'docs/FINAL_CRITERIA.md').write_text(criteria, encoding='utf-8')
    print('Wrote README_FINAL.md, CHANGELOG_FINAL.md and FINAL_CRITERIA.md')


if __name__ == '__main__':
    main()
