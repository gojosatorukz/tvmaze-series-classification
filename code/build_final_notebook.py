"""Create an executable walkthrough of the recorded Final experiment."""
from pathlib import Path
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def md(value):
        cells.append(nbf.v4.new_markdown_cell(value))
    def code(value):
        cells.append(nbf.v4.new_code_cell(value))

    md('''# TVmaze: does a synopsis improve rating classification?

**Bekzat Rashiduly · Machine Learning Algorithms · Final · November 2026**

Research question: does plot-description text improve prediction of **TVmaze rating ≥ 8/10**, given structured metadata?

This notebook is an executable walkthrough of the recorded experiment and the saved-model predictions. It loads the complete search results included with the project. It does **not** silently repeat the expensive exhaustive searches. To retrain every candidate in a new isolated directory, run `python code/run_final.py --fresh --workers 4`. The report documents contributions and tool use.

Data: [TVmaze API](https://www.tvmaze.com/api), CC BY-SA 4.0. Pretrained model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), Apache-2.0.''')
    code('''from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
import joblib
from IPython.display import display, Image

ROOT = Path.cwd()
if not (ROOT / 'code/final_models.py').exists():
    raise RuntimeError('Open this notebook with the tvmaze_project directory as the working directory.')
sys.path.insert(0, str(ROOT / 'code'))
from endterm import check_continuity
from project import FEATURES
from final_models import teacher_grids
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, confusion_matrix

OUT = ROOT / 'results/final'
summary = json.loads((OUT / 'analysis_summary.json').read_text())
_, train, test, folds, midterm, continuity = check_continuity(ROOT)
print('Verified raw response hashes:', continuity['raw_hashes_verified'])
display(pd.DataFrame([
    {'split': 'training', 'series': len(train), 'positive': int(train.high_rated.sum())},
    {'split': 'reused test benchmark', 'series': len(test), 'positive': int(test.high_rated.sum())}
]))''')
    md('''## 1. One series is one example

The collected snapshot contains 14,055 shows. Filtering to rated, English, scripted series leaves 2,339 examples. Unknown ratings cannot provide supervised labels. The seven metadata inputs describe the series; the synopsis is the additional text input. The rating creates the label and is excluded from the model features.

The saved 80/20 split and five outer training folds are the same as earlier course stages. The test set has been seen in earlier reports, so it is a **reused benchmark**, not a fresh prospective evaluation.''')
    code('''print('Metadata features:', FEATURES)
display(train[FEATURES + ['plot_summary']].head(3))
display(train.high_rated.value_counts().rename_axis('rating >= 8').to_frame('training examples'))
assert set(train.show_id).isdisjoint(set(test.show_id))
assert train.plot_summary.str.strip().ne('').all()
assert test.plot_summary.str.strip().ne('').all()''')
    md('''## 2. Feature engineering and model selection

We add genre count, genre combinations, premiere decade and runtime buckets. Numeric missing values use training medians. A separate cross-fitted target-encoding experiment tests sparse categories. Each fitted preprocessing step stays inside the corresponding training fold.

An outer fold evaluates the entire inner tuning procedure: five outer folds, three inner folds. A separate five-fold search on the full training partition chooses the saved settings. Every Cartesian combination below is evaluated; there is no random search.''')
    code('''display(pd.DataFrame([
    {'family': family, 'configurations': len(ParameterGrid(grid)), 'grid': str(grid)}
    for family, (_, grid) in teacher_grids().items()
]))
display(pd.read_csv(OUT / 'parameter_boundaries.csv'))''')
    md('''## 3. Text representations

**TF-IDF** counts words and word pairs, discounts common terms and trains logistic regression on our examples. **LSA** compresses TF-IDF into 48 latent dimensions. **MiniLM** is a frozen pretrained transformer: tokenisation → six transformer layers → masked mean pooling → L2 normalisation → 384 numbers. A logistic classifier learns the rating task from these numbers.

MiniLM uses a pinned local quantised ONNX file and up to 256 tokens per synopsis. We compare metadata, text and both sources on exactly the same objects and folds. The encoder weights remain frozen.''')
    code('''manifest = json.loads((ROOT / 'assets/minilm/manifest.json').read_text())
print('Pinned model revision:', manifest['revision'])
from final_text import LocalEncoder
encoder = LocalEncoder()
texts = train.plot_summary.head(2).tolist()
vectors = encoder.encode(texts)
print('Two descriptions become:', vectors.shape)
print('Vector norms:', np.linalg.norm(vectors, axis=1))
display(pd.DataFrame(vectors[:, :8], columns=[f'dimension {n}' for n in range(8)]))''')
    md('''## 4. Recorded results

Positive-class F1 is the primary metric because only 17.3% of examples have rating ≥8. A model predicting “below 8” for everything gets about 83% accuracy but zero positive recall. Outer-CV mean and standard deviation describe training-fold performance. Test metrics describe the reused 468-object benchmark.

The selection file was written from training-CV scores before this Final test evaluation. Choosing the highest test score after looking at all models would overstate the result.''')
    code('''selection = json.loads((OUT / 'selection_before_test.json').read_text())
print('Primary model selected by training CV:', selection['primary'])
comparison = pd.read_csv(OUT / 'model_comparison.csv')
display(comparison[['name', 'cv_f1_mean', 'cv_f1_sd', 'test_f1', 'test_precision', 'test_recall', 'test_roc_auc']].round(4))
display(Image(filename=str(OUT / 'figures/representation_comparison.png')))
print(summary['conclusion'])
display(pd.DataFrame(summary['deltas'])[['baseline', 'new', 'outer_cv_f1_delta', 'test_f1_delta', 'low_95', 'high_95']].round(4))''')
    md('''## 5. Neural-network checkpoint and overfitting

Four MLP architectures are compared with four L2 penalties and four learning rates. Every fit has its own internal stratified validation split. Validation log loss chooses the epoch; F1 evaluates class predictions. The saved model restores the **absolute minimum validation-loss weights**, including improvements smaller than the stopping tolerance.

Interpret the gap: if fitting loss continues to fall while validation loss rises, the model increasingly fits details that do not generalise to the validation examples. Patience allows observation of this trend; the later weights are discarded.''')
    code('''display(Image(filename=str(OUT / 'figures/architecture_comparison.png')))
display(Image(filename=str(OUT / 'figures/mlp_learning.png')))
mlp = summary['mlp']
display(pd.DataFrame([{key: mlp[key] for key in [
    'best_epoch', 'stopped_epoch', 'best_validation_loss', 'gap_at_best', 'gap_at_stop', 'parameters'
]}]))
history = pd.read_csv(OUT / 'searches/mlp/learning_history.csv')
assert int(history.loc[history.validation_balanced_log_loss.idxmin(), 'epoch']) == mlp['best_epoch']''')
    md('''## 6. Explain specific errors

An error list alone does not explain a model. The reviewed cases below connect observed metadata, training-group statistics and measured model sensitivity to each mistake. Replacing a feature or removing a sentence probes the fitted model. It cannot prove the cause of the audience rating.''')
    code('''findings = json.loads((OUT / 'reviewed_findings.json').read_text())
for case in findings['cases']:
    print(case['name'])
    print(case['text'])
    print()
print(findings['additional_error_note'])
display(pd.read_csv(OUT / 'error_counts.csv'))
display(Image(filename=str(OUT / 'figures/confusion_matrices.png')))''')
    md('''## 7. Actually load a saved model and predict

Training produces a `.joblib` file containing the fitted transformations and classifier. Loading it reconstructs those learned objects. Calling `.predict()` transforms new feature values with the saved preprocessing and runs the learned decision rule. It does **not** retrain the model.

The example below is a real test-series input. Its observed rating is displayed only to assess the prediction. `FEATURES + ['plot_summary']` is the explicit input whitelist.''')
    code('''family = summary['primary']
model = joblib.load(ROOT / 'models/final' / f'{family}.joblib')
example = test.loc[test.show_id.eq(2)].iloc[[0]]
display(example[['name', 'rating'] + FEATURES])
prediction = model.predict(example[FEATURES + ['plot_summary']])
print('Model:', family)
print('Predicted class:', int(prediction[0]), '(1 means rating >= 8)')
all_predictions = model.predict(test[FEATURES + ['plot_summary']])
reported = comparison.set_index('family').loc[family, 'test_f1']
assert np.isclose(f1_score(test.high_rated, all_predictions), reported, atol=1e-12)
print('Reloaded-model test F1:', f1_score(test.high_rated, all_predictions))
print('Confusion matrix [true rows, predicted columns]:')
print(confusion_matrix(test.high_rated, all_predictions))''')
    md('''## 8. A genuinely new input through the app

`python app.py` starts a local server at `http://127.0.0.1:8765`. The browser sends feature values to the Flask route. The route builds an input row, loads the three saved models and returns their outputs. Changing the form changes the input; pressing Predict does not start training.

The following cell calls that same route with a fictional series. Its true rating is unknown. It also demonstrates missing-text fallback and an actionable invalid-input response.''')
    code('''sys.path.insert(0, str(ROOT))
from app import create_app
client = create_app().test_client()
new_series = {
    'premiere_year': 2026, 'runtime_minutes': 45,
    'network': 'Unknown', 'web_channel': 'Netflix',
    'broadcast_country': 'US', 'distribution': 'Streaming',
    'genres': ['Drama', 'Mystery'],
    'plot_summary': 'An astronomer investigates a signal with a retired engineer.'
}
response = client.post('/api/predict', json=new_series)
assert response.status_code == 200, response.json
display(pd.DataFrame(response.json['models']))
missing = client.post('/api/predict', json={**new_series, 'plot_summary': '', 'runtime_minutes': None})
assert missing.status_code == 200
print('Without a synopsis, the available primary role is:', missing.json['primary_role'])
invalid = client.post('/api/predict', json={**new_series, 'runtime_minutes': -10})
assert invalid.status_code == 400
print('Invalid input:', invalid.json['error'])''')
    md('''## 9. What the evidence supports

'''+json.loads((ROOT / 'results/final/reviewed_findings.json').read_text())['conclusion']+'''

The random split can place related franchises in both partitions. The benchmark was reused across course stages. The snapshot excludes unrated series, has no vote counts and covers only rated English scripted series. Synopsis semantics can describe genre without revealing execution quality. A fresh franchise-grouped or temporal evaluation would test a stronger claim about unseen series.

### Reproduce and inspect

```bash
python code/run_final.py --verify-only
python -m unittest discover -s tests -v
python app.py
# Expensive: rerun every saved full grid
python code/run_final.py --fresh --workers 4
```

The report contains parameter ranges, positions of selected settings, specific error explanations, feedback responses and required tool disclosure. Full grids are in `results/final/searches/`; per-series predictions and paired intervals are in `results/final/`.''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3.12 (TVmaze)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.14'},
    })
    path = ROOT / 'Final_TVSeries.ipynb'
    nbf.write(notebook, path)
    print(path)


if __name__ == '__main__':
    main()
