"""Model definitions and feature transformations for the Final experiments."""

import copy
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from endterm import BalancedBoosting, CheckedMLP
from project import CATEGORICAL, FEATURES, NUMERIC, SEED, GenreEncoder


class DerivedMetadata(TransformerMixin, BaseEstimator):
    """Derive coarse bins and interactions from the registered input fields."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()
        genres = frame.genres.map(json.loads)
        frame['genre_count'] = genres.map(len)
        frame['genre_combination'] = genres.map(lambda values: '|'.join(sorted(values)) or 'Unknown')
        frame['premiere_decade'] = frame.premiere_year.map(
            lambda year: str(int(year) // 10 * 10) if pd.notna(year) else 'Unknown')
        frame['runtime_bucket'] = pd.cut(
            frame.runtime_minutes, [0, 20, 40, 65, np.inf],
            labels=['up_to_20', '21_to_40', '41_to_65', 'over_65'],
        ).astype(object).fillna('Unknown')
        return frame


def metadata(scale=False, engineered=True):
    numeric = NUMERIC + (['genre_count'] if engineered else [])
    categories = CATEGORICAL + (['genre_combination', 'premiere_decade', 'runtime_bucket'] if engineered else [])
    numeric_steps = [('impute', SimpleImputer(strategy='median', add_indicator=True))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    columns = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), numeric),
        ('categories', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10,
                                     sparse_output=False), categories),
        ('genres', GenreEncoder(), ['genres']),
    ])
    steps = [('derive', DerivedMetadata())] if engineered else []
    return Pipeline(steps + [('encode', columns)])


class CheckpointMLP(CheckedMLP):
    """Select absolute minimum validation log loss and restore its model weights."""

    def fit(self, X, y):
        labels = np.asarray(y)
        fit_ids, val_ids = train_test_split(
            np.arange(len(labels)), test_size=self.validation_fraction,
            stratify=labels, random_state=self.random_state)
        self.prepare_ = metadata(scale=True)
        fitting = self.prepare_.fit_transform(X.iloc[fit_ids], labels[fit_ids])
        validation = self.prepare_.transform(X.iloc[val_ids])
        fit_labels, val_labels = labels[fit_ids], labels[val_ids]
        fit_weights = compute_sample_weight('balanced', fit_labels)
        val_weights = compute_sample_weight('balanced', val_labels)
        self.calibration_train_size_ = len(fit_ids)
        self.calibration_validation_size_ = len(val_ids)
        self.history_ = []
        self.best_validation_loss_ = float('inf')
        significant_best = float('inf')
        stale = 0
        model = self._new_model()
        for epoch in range(1, self.max_epochs + 1):
            model.partial_fit(fitting, fit_labels, classes=np.array([0, 1]), sample_weight=fit_weights)
            fit_proba = model.predict_proba(fitting)
            val_proba = model.predict_proba(validation)
            val_loss = log_loss(val_labels, val_proba, sample_weight=val_weights, labels=[0, 1])
            self.history_.append({
                'epoch': epoch,
                'train_balanced_log_loss': log_loss(fit_labels, fit_proba, sample_weight=fit_weights, labels=[0, 1]),
                'validation_balanced_log_loss': val_loss,
                'train_f1': f1_score(fit_labels, fit_proba[:, 1] >= .5, zero_division=0),
                'validation_f1': f1_score(val_labels, val_proba[:, 1] >= .5, zero_division=0),
            })
            if val_loss < self.best_validation_loss_:
                self.best_validation_loss_ = val_loss
                self.best_epoch_ = epoch
                self.model_ = copy.deepcopy(model)
            if val_loss < significant_best - self.min_delta:
                significant_best, stale = val_loss, 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        self.stopped_epoch_ = epoch
        self.stop_reason_ = 'patience' if stale >= self.patience else 'epoch cap'
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        restored_loss = log_loss(val_labels, self.predict_proba(X.iloc[val_ids]),
                                 sample_weight=val_weights, labels=[0, 1])
        if not np.isclose(restored_loss, self.best_validation_loss_, atol=1e-12, rtol=0):
            raise RuntimeError('Restored checkpoint differs from the recorded validation minimum.')
        return self


def teacher_grids():
    return {
        'random_forest': (
            Pipeline([('prepare', metadata()), ('model', RandomForestClassifier(
                n_estimators=200, class_weight='balanced', random_state=SEED, n_jobs=1))]),
            {'model__n_estimators': [100, 200, 400, 600],
             'model__max_depth': [4, 8, 16, None], 'model__min_samples_leaf': [2, 5, 10, 20],
             'model__max_features': [0.1, 0.25, 0.5, 1.0]},
        ),
        'gradient_boosting': (
            Pipeline([('prepare', metadata()), ('model', BalancedBoosting(
                random_state=SEED, min_samples_leaf=10))]),
            {'model__n_estimators': [50, 100, 200, 300], 'model__learning_rate': [.03, .06, .1, .2],
             'model__max_depth': [1, 2, 3, 4], 'model__min_samples_leaf': [2, 5, 10, 20]},
        ),
        'mlp': (
            CheckpointMLP(max_epochs=150, patience=15, min_delta=.0001),
            {'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64)],
             'alpha': [.001, .01, .05, .2], 'learning_rate_init': [.0003, .001, .003, .01]},
        ),
    }


def synopsis_pipeline(use_metadata, representation='none', engineered=True):
    blocks = []
    if use_metadata:
        blocks.append(('metadata', metadata(scale=True, engineered=engineered), FEATURES))
    if representation != 'none':
        vectorizer = TfidfVectorizer(
            stop_words='english', strip_accents='unicode', ngram_range=(1, 2),
            min_df=2, max_df=.95, sublinear_tf=True, max_features=20000)
        if representation == 'lsa':
            vectorizer = Pipeline([('tfidf', vectorizer),
                                   ('svd', TruncatedSVD(n_components=48, n_iter=5, random_state=SEED)),
                                   ('normalize', Normalizer())])
        blocks.append(('text', vectorizer, 'plot_summary'))
    encoder = ColumnTransformer(blocks, sparse_threshold=1.0)
    model = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', max_iter=3000,
                               random_state=SEED)
    return Pipeline([('prepare', encoder), ('model', model)])


def synopsis_grids():
    choices = {}
    for name, use_metadata, representation, engineered in [
        ('original_metadata', True, 'none', False),
        ('engineered_metadata', True, 'none', True),
        ('tfidf_text', False, 'tfidf', True),
        ('tfidf_combined', True, 'tfidf', True),
        ('lsa_text', False, 'lsa', True),
        ('lsa_combined', True, 'lsa', True),
    ]:
        grid = {'model__C': [.001, .01, .1, 1]}
        if use_metadata and representation != 'none':
            grid['prepare__transformer_weights'] = [
                {'metadata': 1, 'text': weight} for weight in [.3, 1, 3, 6]]
        choices[name] = (synopsis_pipeline(use_metadata, representation, engineered), grid)
    return choices
