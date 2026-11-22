"""Freeze training-only choices, evaluate the saved benchmark and export evidence."""

import os
from pathlib import Path
import json
import hashlib
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/final'
FIG = OUT / 'figures'
MODELS = ROOT / 'models/final'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.runtime/final-matplotlib')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from endterm import check_continuity, fixed_cv, scores
from final_models import teacher_grids, synopsis_grids
from final_text import semantic_grids
from target_features import target_grids
from final_experiments import save_json
from project import FEATURES, NUMERIC, make_models, positive_scores

NAMES = {
    'random_forest': 'Random Forest, full grid', 'gradient_boosting': 'Gradient Boosting, full grid',
    'mlp': 'MLP, architecture search', 'original_metadata': 'Original metadata + LR',
    'engineered_metadata': 'Engineered metadata + LR', 'tfidf_text': 'TF-IDF text + LR',
    'tfidf_combined': 'Metadata + TF-IDF + LR', 'lsa_text': 'LSA text + LR',
    'lsa_combined': 'Metadata + LSA + LR', 'minilm_text': 'MiniLM text + LR',
    'minilm_combined': 'Metadata + MiniLM + LR', 'svm_reference': 'Midterm SVM reference',
    'endterm_rf_reference': 'Endterm Random Forest reference',
    'dummy_majority': 'Majority baseline', 'dummy_stratified': 'Stratified random baseline',
    'target_encoded_metadata': 'Cross-fitted target encoding + LR',
    'target_minilm_combined': 'Target encoding + MiniLM + LR',
}
METADATA = ['random_forest', 'gradient_boosting', 'mlp', 'original_metadata',
            'engineered_metadata', 'svm_reference', 'endterm_rf_reference', 'target_encoded_metadata']
TEXT = ['tfidf_text', 'lsa_text', 'minilm_text']
COMBINED = ['tfidf_combined', 'lsa_combined', 'minilm_combined', 'target_minilm_combined']


def diagnostic_score(model, frame):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(frame)[:, list(model.classes_).index(1)]
    return model.decision_function(frame)


def bootstrap_delta(actual, base, new, repeats=2000):
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(actual == label) for label in [0, 1]]
    samples = np.concatenate([rng.choice(group, size=(repeats, len(group)), replace=True)
                              for group in groups], axis=1)
    labels = actual[samples]
    def f1_values(prediction):
        prediction = prediction[samples]
        tp = ((prediction == 1) & (labels == 1)).sum(axis=1)
        fp = ((prediction == 1) & (labels == 0)).sum(axis=1)
        fn = ((prediction == 0) & (labels == 1)).sum(axis=1)
        return 2*tp / np.maximum(2*tp+fp+fn, 1)
    differences = f1_values(new) - f1_values(base)
    return {'bootstrap_repeats': repeats, 'low_95': float(np.quantile(differences, .025)),
            'high_95': float(np.quantile(differences, .975)),
            'method': 'Paired stratified percentile bootstrap of fixed test objects, seed 42',
            'scope': 'Conditional on fitted models and this benchmark. Retraining and franchise dependence are not represented.'}


def parameter_boundaries(definitions):
    records = []
    for name in ['random_forest', 'gradient_boosting', 'mlp']:
        folder = OUT / 'searches' / name
        selection = json.loads((folder / 'full_train.json').read_text())
        grid = definitions[name][1]
        frame = pd.read_csv(folder / 'full_train_grid.csv')
        if len(frame) != len(ParameterGrid(grid)):
            raise ValueError(f'{name}: incomplete Cartesian grid.')
        for parameter, values in grid.items():
            best = selection['best_params'][parameter]
            normal = [list(value) if isinstance(value, tuple) else value for value in values]
            index = normal.index(best)
            if parameter == 'hidden_layer_sizes':
                position = f'architecture {index+1} of {len(values)}'
            elif best is None:
                position = 'unbounded-depth option'
            elif index == 0:
                position = 'lower boundary'
            elif index == len(values)-1:
                position = 'upper boundary'
            else:
                position = 'interior'
            records.append({'family': name, 'parameter': parameter, 'tested': json.dumps(normal),
                            'selected': json.dumps(best), 'position': position})
    pd.DataFrame(records).to_csv(OUT / 'parameter_boundaries.csv', index=False)
    return records


def case_evidence(train, test, predictions, fitted, scores_by_model, best_metadata, best_combined):
    actual = test.high_rated.to_numpy()
    meta_wrong = predictions[best_metadata] != actual
    combo_wrong = predictions[best_combined] != actual
    selected_ids = [2, 185, 318]
    for mask in [meta_wrong & ~combo_wrong, ~meta_wrong & combo_wrong, meta_wrong & combo_wrong]:
        selected_ids.extend(test.loc[mask].sort_values('show_id').head(2).show_id.tolist())
    control_wrong = predictions['engineered_metadata'] != actual
    for mask in [control_wrong & ~combo_wrong, ~control_wrong & combo_wrong]:
        selected_ids.extend(test.loc[mask].sort_values('show_id').head(2).show_id.tolist())
    records = []
    relevant = list(dict.fromkeys([best_metadata, best_combined, 'engineered_metadata', 'minilm_text', 'minilm_combined',
                                  'tfidf_combined', 'svm_reference']))
    defaults = {column: train[column].median() if column in NUMERIC else train[column].mode().iloc[0]
                for column in FEATURES}
    for show_id in dict.fromkeys(selected_ids):
        position = np.flatnonzero(test.show_id.to_numpy() == show_id)
        if not len(position):
            continue
        index = int(position[0])
        row = test.iloc[index]
        X = test.iloc[[index]][FEATURES+['plot_summary']]
        details = {'show_id': int(show_id), 'name': row['name'], 'rating': float(row.rating),
                   'actual': int(row.high_rated), 'features': {key: row[key] for key in FEATURES},
                   'synopsis': row.plot_summary, 'models': {}, 'metadata_sensitivity': [],
                   'sentence_sensitivity': [], 'training_groups': []}
        for family in relevant:
            details['models'][family] = {'predicted': int(predictions[family][index]),
                                         'score': float(scores_by_model[family][index]),
                                         'threshold': .5 if hasattr(fitted[family], 'predict_proba') else 0,
                                         'score_kind': 'model probability score' if hasattr(fitted[family], 'predict_proba') else 'SVM margin'}
        baseline = float(scores_by_model[best_metadata][index])
        for feature, replacement in defaults.items():
            variant = X.copy()
            variant.loc[:, feature] = replacement
            modified = float(diagnostic_score(fitted[best_metadata], variant)[0])
            details['metadata_sensitivity'].append({'feature': feature, 'observed': row[feature],
                                                     'training_reference': replacement,
                                                     'original_score': baseline, 'replacement_score': modified,
                                                     'delta': baseline-modified})
        details['metadata_sensitivity'].sort(key=lambda item: abs(item['delta']), reverse=True)
        details['controlled_metadata_sensitivity'] = []
        baseline = float(scores_by_model['engineered_metadata'][index])
        for feature, replacement in defaults.items():
            variant = X.copy()
            variant.loc[:, feature] = replacement
            modified = float(diagnostic_score(fitted['engineered_metadata'], variant)[0])
            details['controlled_metadata_sensitivity'].append({
                'feature': feature, 'observed': row[feature], 'training_reference': replacement,
                'original_score': baseline, 'replacement_score': modified, 'delta': baseline-modified})
        details['controlled_metadata_sensitivity'].sort(key=lambda item: abs(item['delta']), reverse=True)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', row.plot_summary.strip())
        if len(sentences) > 1:
            for sentence in sentences[:5]:
                variant = X.copy()
                variant.loc[:, 'plot_summary'] = row.plot_summary.replace(sentence, '', 1).strip()
                modified = float(diagnostic_score(fitted[best_combined], variant)[0])
                initial = float(scores_by_model[best_combined][index])
                details['sentence_sensitivity'].append({'removed': sentence, 'original_score': initial,
                                                         'replacement_score': modified, 'delta': initial-modified})
            details['sentence_sensitivity'].sort(key=lambda item: abs(item['delta']), reverse=True)
        for feature in ['network', 'genres']:
            group = train.loc[train[feature].eq(row[feature])]
            details['training_groups'].append({'feature': feature, 'value': row[feature],
                                               'training_rows': len(group), 'positives': int(group.high_rated.sum()),
                                               'positive_share': float(group.high_rated.mean()) if len(group) else None})
        records.append(details)
    save_json(OUT / 'error_evidence.json', records)
    return records


def make_figures(comparison, best_metadata, best_text, best_combined, actual, predictions):
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
    selected = ['engineered_metadata', 'tfidf_text', 'tfidf_combined', 'minilm_text', 'minilm_combined', best_metadata]
    selected = list(dict.fromkeys(selected))
    data = comparison.set_index('family').loc[selected]
    fig, ax = plt.subplots(figsize=(10, 4.9))
    positions = np.arange(len(data))
    ax.barh(positions-.18, data.cv_f1_mean, .34, xerr=data.cv_f1_sd, label='Outer CV mean ± SD', color='#0b8178', capsize=3)
    ax.barh(positions+.18, data.test_f1, .34, label='Reused test benchmark', color='#203d50')
    ax.set(yticks=positions, yticklabels=[NAMES[key] for key in selected], xlim=(0, .65), xlabel='Positive-class F1')
    ax.invert_yaxis()
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG / 'representation_comparison.png', bbox_inches='tight')
    plt.close(fig)
    history = pd.read_csv(OUT / 'searches/mlp/learning_history.csv')
    selection = json.loads((OUT / 'searches/mlp/full_train.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for prefix, label, color in [('train', 'Fitting', '#203d50'), ('validation', 'Validation', '#cb6247')]:
        axes[0].plot(history.epoch, history[f'{prefix}_balanced_log_loss'], label=label, color=color)
        axes[1].plot(history.epoch, history[f'{prefix}_f1'], label=label, color=color)
    for ax in axes:
        ax.axvline(selection['best_epoch'], color='#0b8178', linestyle='--', label='Restored checkpoint')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=8)
    axes[0].set(ylabel='Balanced log loss', title='Checkpoint selected by validation loss')
    axes[1].set(ylabel='Positive-class F1', title='F1 tracked during the same run')
    fig.tight_layout()
    fig.savefig(FIG / 'mlp_learning.png', bbox_inches='tight')
    plt.close(fig)
    grid = pd.read_csv(OUT / 'searches/mlp/full_train_grid.csv')
    indices = grid.groupby('param_hidden_layer_sizes').mean_test_score.idxmax()
    architecture = grid.loc[indices].sort_values('mean_test_score', ascending=False)
    architecture.to_csv(OUT / 'architecture_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.bar(architecture.param_hidden_layer_sizes.astype(str), architecture.mean_test_score,
           yerr=architecture.std_test_score, capsize=5, color='#0b8178')
    ax.set(xlabel='Hidden-layer architecture', ylabel='Five-fold selection F1', ylim=(0, .65),
           title='Best grid configuration within each architecture')
    fig.tight_layout()
    fig.savefig(FIG / 'architecture_comparison.png', bbox_inches='tight')
    plt.close(fig)
    families = list(dict.fromkeys([best_metadata, best_text, best_combined]))
    fig, axes = plt.subplots(1, len(families), figsize=(4.1*len(families), 3.8))
    for ax, family in zip(np.atleast_1d(axes), families):
        ConfusionMatrixDisplay.from_predictions(actual, predictions[family], display_labels=['Below 8', 'At least 8'],
                                                 colorbar=False, cmap='Blues', ax=ax)
        ax.set_title(NAMES[family], fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / 'confusion_matrices.png', bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    definitions = {**teacher_grids(), **synopsis_grids(), **semantic_grids(), **target_grids()}
    missing = [name for name in definitions if not (OUT / 'searches' / name / 'full_train.json').exists()]
    if missing:
        raise RuntimeError('Searches still pending: '+', '.join(missing))
    _, train, test, folds, midterm, continuity = check_continuity(ROOT)
    summary = []
    fitted = {}
    for name in definitions:
        folder = OUT / 'searches' / name
        records = pd.read_csv(folder / 'outer_scores.csv')
        if len(records) != 5:
            raise ValueError(f'{name}: expected five outer folds.')
        summary.append({'family': name, 'name': NAMES[name], 'cv_f1_mean': records.f1.mean(),
                        'cv_f1_sd': records.f1.std(ddof=1), 'cv_roc_auc_mean': records.roc_auc.mean()})
        fitted[name] = joblib.load(MODELS / f'{name}.joblib')
    controls = make_models()
    for family, original in [('svm_reference', 'SVM'), ('dummy_majority', 'Dummy majority'),
                              ('dummy_stratified', 'Dummy stratified')]:
        rows = pd.DataFrame(fixed_cv(family, controls[original], train, folds))
        summary.append({'family': family, 'name': NAMES[family], 'cv_f1_mean': rows.f1.mean(),
                        'cv_f1_sd': rows.f1.std(ddof=1), 'cv_roc_auc_mean': rows.roc_auc.mean()})
        model_path = MODELS / f'{family}.joblib'
        if model_path.exists():
            fitted[family] = joblib.load(model_path)
        else:
            fitted[family] = clone(controls[original]).fit(train[FEATURES], train.high_rated)
            joblib.dump(fitted[family], model_path)
    old = json.loads((ROOT / 'results/endterm/experiment_summary.json').read_text())
    reference = next(record for record in old['models'] if record['model'] == 'Random Forest')
    reference_path = MODELS / 'endterm_rf_reference.joblib'
    if not reference_path.exists():
        shutil.copy2(ROOT / 'models/endterm_model.joblib', reference_path)
    fitted['endterm_rf_reference'] = joblib.load(reference_path)
    summary.append({'family': 'endterm_rf_reference', 'name': NAMES['endterm_rf_reference'],
                    'cv_f1_mean': reference['cv_f1_mean'], 'cv_f1_sd': reference['cv_f1_sd'],
                    'cv_roc_auc_mean': reference['cv_roc_auc_mean']})
    ranking = pd.DataFrame(summary).sort_values(['cv_f1_mean', 'cv_roc_auc_mean', 'family'],
                                              ascending=[False, False, True])
    def best(candidates):
        return ranking.loc[ranking.family.isin(candidates)].iloc[0].family
    best_metadata, best_text, best_combined = best(METADATA), best(TEXT), best(COMBINED)
    primary = best(METADATA+TEXT+COMBINED)
    manifest = json.loads((ROOT / 'assets/minilm/manifest.json').read_text())
    selection = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'primary': primary,
        'best_metadata': best_metadata, 'best_text': best_text, 'best_combined': best_combined,
        'metric': 'Mean outer training CV positive-class F1', 'test_used_for_selection': False,
        'target_threshold': 8, 'cv_rankings': ranking.to_dict('records'),
        'continuity': continuity, 'minilm_revision': manifest['revision'],
        'benchmark_context': 'Same test objects already reported at Midterm and Endterm.',
        'model_sha256': {family: hashlib.sha256((MODELS / f'{family}.joblib').read_bytes()).hexdigest()
                         for family in fitted},
    }
    selection_path = OUT / 'selection_before_test.json'
    if selection_path.exists():
        previous = json.loads(selection_path.read_text())
        if any(previous.get(key) != value for key, value in selection.items() if key != 'created_utc'):
            raise RuntimeError('Frozen choices or fitted models changed. Use a new experiment directory.')
    else:
        save_json(selection_path, selection)
    predictions, ranking_scores, diagnostic_scores, test_rows = {}, {}, {}, []
    X_test, actual = test[FEATURES+['plot_summary']], test.high_rated.to_numpy()
    for family, model in fitted.items():
        predictions[family] = model.predict(X_test)
        ranking_scores[family] = positive_scores(model, X_test)
        diagnostic_scores[family] = diagnostic_score(model, X_test)
        test_rows.append({'family': family, **{'test_'+key: value for key, value in scores(
            actual, predictions[family], ranking_scores[family]).items()}})
    comparison = ranking.merge(pd.DataFrame(test_rows), on='family', validate='one_to_one')
    comparison.to_csv(OUT / 'model_comparison.csv', index=False)
    audit = test[['show_id', 'name', 'rating', 'high_rated']].copy()
    errors = []
    for family, predicted in predictions.items():
        audit[family] = predicted
        audit[family+'_score'] = diagnostic_scores[family]
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        errors.append({'family': family, 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)})
    audit.to_csv(OUT / 'test_predictions.csv', index=False)
    pd.DataFrame(errors).to_csv(OUT / 'error_counts.csv', index=False)
    deltas = []
    indexed = comparison.set_index('family')
    for baseline in ['engineered_metadata', best_metadata]:
        for new in ['tfidf_combined', 'lsa_combined', 'minilm_combined']:
            deltas.append({'baseline': baseline, 'new': new,
                           'test_f1_delta': float(indexed.loc[new, 'test_f1']-indexed.loc[baseline, 'test_f1']),
                           'outer_cv_f1_delta': float(indexed.loc[new, 'cv_f1_mean']-indexed.loc[baseline, 'cv_f1_mean']),
                           **bootstrap_delta(actual, predictions[baseline], predictions[new])})
    deltas.append({'baseline': 'target_encoded_metadata', 'new': 'target_minilm_combined',
                   'test_f1_delta': float(indexed.loc['target_minilm_combined', 'test_f1']-indexed.loc['target_encoded_metadata', 'test_f1']),
                   'outer_cv_f1_delta': float(indexed.loc['target_minilm_combined', 'cv_f1_mean']-indexed.loc['target_encoded_metadata', 'cv_f1_mean']),
                   **bootstrap_delta(actual, predictions['target_encoded_metadata'], predictions['target_minilm_combined'])})
    save_json(OUT / 'paired_deltas.json', deltas)
    boundaries = parameter_boundaries(definitions)
    examples = case_evidence(train, test, predictions, fitted, diagnostic_scores, best_metadata, best_combined)
    make_figures(comparison, best_metadata, best_text, best_combined, actual, predictions)
    history = pd.read_csv(OUT / 'searches/mlp/learning_history.csv')
    mlp = json.loads((OUT / 'searches/mlp/full_train.json').read_text())
    if int(history.loc[history.validation_balanced_log_loss.idxmin(), 'epoch']) != mlp['best_epoch']:
        raise ValueError('MLP checkpoint is not the absolute validation-loss minimum.')
    chosen = history.loc[history.epoch.eq(mlp['best_epoch'])].iloc[0]
    last = history.iloc[-1]
    mlp['gap_at_best'] = float(chosen.validation_balanced_log_loss-chosen.train_balanced_log_loss)
    mlp['gap_at_stop'] = float(last.validation_balanced_log_loss-last.train_balanced_log_loss)
    mlp['last_train_loss'] = float(last.train_balanced_log_loss)
    mlp['last_validation_loss'] = float(last.validation_balanced_log_loss)
    mlp['best_train_loss'] = float(chosen.train_balanced_log_loss)
    mlp['parameters'] = int(sum(value.size for value in fitted['mlp'].model_.coefs_)+
                            sum(value.size for value in fitted['mlp'].model_.intercepts_))
    mlp['encoded_inputs'] = int(fitted['mlp'].model_.coefs_[0].shape[0])
    by_role = {'metadata': best_metadata, 'text': best_text, 'combined': best_combined}
    role = next(key for key, family in by_role.items() if family == primary)
    minilm_delta = next(item for item in deltas if item['baseline']=='engineered_metadata' and item['new']=='minilm_combined')
    direction = 'increased' if minilm_delta['test_f1_delta'] > 0 else 'decreased'
    conclusion = (f'MiniLM added to the same metadata logistic-regression pipeline {direction} test F1 '
                  f'by {abs(minilm_delta["test_f1_delta"]):.3f}. The paired 95% bootstrap interval is '
                  f'[{minilm_delta["low_95"]:.3f}, {minilm_delta["high_95"]:.3f}]. '
                  f'Training cross-validation selected {NAMES[primary]} as the primary model.')
    app_manifest = {'primary_family': primary, 'primary_role': role, 'conclusion': conclusion,
                    'controlled_metadata': {'family': 'engineered_metadata',
                                             'cv_f1': float(indexed.loc['engineered_metadata', 'cv_f1_mean']),
                                             'test_f1': float(indexed.loc['engineered_metadata', 'test_f1'])},
                    'models': {key: {'family': family, 'cv_f1': float(indexed.loc[family, 'cv_f1_mean']),
                                      'test_f1': float(indexed.loc[family, 'test_f1'])}
                               for key, family in by_role.items()},
                    'source': 'TVmaze, CC BY-SA 4.0', 'cohort': 2339, 'test_rows': 468}
    save_json(OUT / 'app_manifest.json', app_manifest)
    save_json(OUT / 'analysis_summary.json', {
        'primary': primary, 'best_metadata': best_metadata, 'best_text': best_text,
        'best_combined': best_combined, 'models': comparison.to_dict('records'),
        'deltas': deltas, 'mlp': mlp, 'parameter_boundaries': boundaries, 'errors': errors,
        'examples': examples, 'conclusion': conclusion, 'continuity': continuity,
        'pretrained_model': {'id': manifest['model_id'], 'revision': manifest['revision'],
                             'dimensions': 384, 'max_tokens': 256, 'license': 'Apache-2.0'},
    })
    print(comparison[['family', 'cv_f1_mean', 'cv_f1_sd', 'test_f1']].round(4).to_string(index=False))
    print(conclusion)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
