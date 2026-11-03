"""Endterm experiments. All selection and learned transforms use training rows only.

The original project.py and its data/split files remain unchanged.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, silhouette_score, adjusted_rand_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from project import (FEATURES, SEED, THRESHOLD, clean_raw, freeze_splits, load_raw,
                     make_models, positive_scores, preprocessing)


class BalancedBoosting(GradientBoostingClassifier):
    """Recompute class-balanced sample weights inside each fit, including inner CV."""
    def fit(self, X, y, sample_weight=None, monitor=None):
        weights = compute_sample_weight('balanced', y)
        if sample_weight is not None:
            weights = weights * sample_weight
        return super().fit(X, y, sample_weight=weights, monitor=monitor)


class CheckedMLP(ClassifierMixin, BaseEstimator):
    """Select epoch using an inner validation loss, then refit on all supplied rows.

    The calibration preprocessor sees only the inner fitting rows. The validation
    and outer test rows never enter its fit. Final preprocessing uses all supplied
    outer-training rows, and the fresh MLP trains for the selected epoch count.
    """
    def __init__(self, hidden_layer_sizes=(64, 32), alpha=0.01,
                 learning_rate_init=0.001, max_epochs=180, patience=20,
                 min_delta=0.0001, validation_fraction=0.2, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _new_model(self):
        return MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation='relu',
                             solver='adam', alpha=self.alpha, batch_size=64,
                             learning_rate_init=self.learning_rate_init, max_iter=1,
                             shuffle=True, random_state=self.random_state)

    def fit(self, X, y):
        y = np.asarray(y)
        fit_ids, val_ids = train_test_split(np.arange(len(y)), test_size=self.validation_fraction,
                                          stratify=y, random_state=self.random_state)
        self.calibration_train_size_ = len(fit_ids)
        self.calibration_validation_size_ = len(val_ids)
        prep = preprocessing(True)
        A = prep.fit_transform(X.iloc[fit_ids], y[fit_ids])
        B = prep.transform(X.iloc[val_ids])
        weights = compute_sample_weight('balanced', y[fit_ids])
        va_weights = compute_sample_weight('balanced', y[val_ids])
        model = self._new_model()
        self.history_ = []
        best, best_epoch, stale = float('inf'), 1, 0
        for epoch in range(1, self.max_epochs + 1):
            model.partial_fit(A, y[fit_ids], classes=np.array([0, 1]), sample_weight=weights)
            pa, pb = model.predict_proba(A), model.predict_proba(B)
            train_loss = log_loss(y[fit_ids], pa, sample_weight=weights, labels=[0, 1])
            val_loss = log_loss(y[val_ids], pb, sample_weight=va_weights, labels=[0, 1])
            self.history_.append({'epoch':epoch, 'train_balanced_log_loss':train_loss,
                                  'validation_balanced_log_loss':val_loss,
                                  'train_f1':f1_score(y[fit_ids],pa[:,1]>=.5,zero_division=0),
                                  'validation_f1':f1_score(y[val_ids],pb[:,1]>=.5,zero_division=0)})
            if val_loss < best - self.min_delta:
                best, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        self.best_epoch_ = best_epoch
        self.stopped_epoch_ = epoch
        self.best_validation_loss_ = best
        self.stop_reason_ = 'validation plateau' if stale >= self.patience else 'epoch cap'
        self.prepare_ = preprocessing(True)
        Z = self.prepare_.fit_transform(X, y)
        self.model_ = self._new_model()
        weights = compute_sample_weight('balanced', y)
        for _ in range(self.best_epoch_):
            self.model_.partial_fit(Z, y, classes=np.array([0, 1]), sample_weight=weights)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.prepare_.transform(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def scores(y, predicted, ranking):
    return {'f1':f1_score(y,predicted,zero_division=0),
            'precision':precision_score(y,predicted,zero_division=0),
            'recall':recall_score(y,predicted,zero_division=0),
            'roc_auc':roc_auc_score(y,ranking),
            'average_precision':average_precision_score(y,ranking),
            'balanced_accuracy':balanced_accuracy_score(y,predicted),
            'accuracy':accuracy_score(y,predicted)}


def study_definitions():
    """Finite, declared search spaces. No study depends on held-out results."""
    forest = Pipeline([('prepare',preprocessing(False)),
                       ('model',RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                       random_state=SEED, n_jobs=1))])
    boost = Pipeline([('prepare',preprocessing(False)),
                      ('model',BalancedBoosting(random_state=SEED, min_samples_leaf=10))])
    spaces = {
        'Random Forest':(forest, {'model__max_depth':[8,None],
                                 'model__min_samples_leaf':[2,10],
                                 'model__max_features':['sqrt',.5]}),
        'Gradient Boosting':(boost, {'model__n_estimators':[100,200],
                                    'model__learning_rate':[.03,.1],
                                    'model__max_depth':[1,2]})}
    pca_svm = Pipeline([('prepare',preprocessing(True)),
                        ('pca',PCA(n_components=.9,svd_solver='full')),
                        ('model',SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced'))])
    return spaces, pca_svm, CheckedMLP()


def fixed_cv(name, model, train, folds):
    records = []
    X_all, y_all = train[FEATURES], train.high_rated
    for i,(a,b) in enumerate(folds,1):
        fitted = clone(model).fit(X_all.iloc[a],y_all.iloc[a])
        X = X_all.iloc[b]
        row = {'model':name,'fold':i, **scores(train.iloc[b].high_rated,
                                               fitted.predict(X),positive_scores(fitted,X))}
        if isinstance(fitted,CheckedMLP):
            row.update(best_epoch=fitted.best_epoch_,stopped_epoch=fitted.stopped_epoch_)
        records.append(row)
    return records


def search(estimator, space, train, folds):
    gs = GridSearchCV(estimator,space,scoring='f1',cv=folds,n_jobs=2,
                      error_score='raise',refit=True,return_train_score=True)
    with parallel_backend('threading'), threadpool_limits(limits=1):
        gs.fit(train[FEATURES],train.high_rated)
    return gs


def tune_ensemble(name, estimator, space, train, outer_folds, out):
    """Independent inner search per frozen outer fold, then a full-train search."""
    records = []
    for i,(a,b) in enumerate(outer_folds,1):
        print(f'{name}: nested outer fold {i}/5',flush=True)
        inner = StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
        fitted = search(estimator,space,train.iloc[a].reset_index(drop=True),inner)
        X = train.iloc[b][FEATURES]
        records.append({'model':name,'fold':i,
                        **scores(train.iloc[b].high_rated,fitted.predict(X),positive_scores(fitted.best_estimator_,X)),
                        'inner_best_f1':fitted.best_score_,'selected_params':json.dumps(fitted.best_params_,sort_keys=True)})
    print(f'{name}: full-training search',flush=True)
    full = search(estimator,space,train,outer_folds)
    pd.DataFrame(full.cv_results_).to_csv(out/(name.lower().replace(' ','_')+'_search.csv'),index=False)
    return records, full


def check_continuity(root):
    raw, manifest = load_raw(root)
    clean, audit = clean_raw(raw,manifest['collection_date_almaty'])
    old = json.loads((root/'results/midterm/experiment_summary.json').read_text())
    clean_bytes = (root/'data/processed/series_clean.csv').read_bytes()
    assert hashlib.sha256(clean_bytes).hexdigest() == old['clean_csv_sha256']
    assert hashlib.sha256((root/'data/splits/split_assignments.csv').read_bytes()).hexdigest() == old['split_sha256']
    pd.testing.assert_frame_equal(clean,pd.read_csv(root/'data/processed/series_clean.csv',keep_default_na=False,
                                                   na_values={'runtime_minutes':[''],'premiere_year':['']}),check_dtype=False)
    train,test,folds = freeze_splits(clean,root)
    evidence = {'raw_hashes_verified':len(manifest['pages']), 'clean_rows':len(clean),
                'train_rows':len(train),'test_rows':len(test),'positive_rows':int(clean.high_rated.sum()),
                'clean_csv_sha256':old['clean_csv_sha256'],'split_sha256':old['split_sha256'],
                'same_cohort_labels_features_split':True,'target_threshold':THRESHOLD}
    return clean,train,test,folds,old,evidence


def unsupervised_analysis(train,out):
    """Fit on training objects, use labels only afterwards for description."""
    prep = preprocessing(True)
    Z = prep.fit_transform(train[FEATURES])
    names = prep.get_feature_names_out()
    pca = PCA(n_components=.9,svd_solver='full').fit(Z)
    P = pca.transform(Z)
    loadings = pd.DataFrame(pca.components_[:2].T,index=names,columns=['PC1','PC2'])
    loadings.to_csv(out/'pca_loadings.csv')
    variance = pd.DataFrame({'component':np.arange(1,len(pca.explained_variance_ratio_)+1),
                            'explained_variance':pca.explained_variance_ratio_,
                            'cumulative':pca.explained_variance_ratio_.cumsum()})
    variance.to_csv(out/'pca_variance.csv',index=False)
    rows, fitted = [], {}
    for k in range(2,7):
        km = KMeans(n_clusters=k,n_init=20,random_state=SEED).fit(P)
        rows.append({'k':k,'silhouette':silhouette_score(P,km.labels_), 'inertia':km.inertia_})
        fitted[k] = km
    quality = pd.DataFrame(rows)
    quality.to_csv(out/'cluster_search.csv',index=False)
    k = int(quality.sort_values(['silhouette','k'],ascending=[False,True]).iloc[0].k)
    km = fitted[k]
    stability=[]
    for seed in [7,21,84]:
        alt=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(P)
        stability.append({'seed':seed,'adjusted_rand_index':adjusted_rand_score(km.labels_,alt.labels_)})
    stability=pd.DataFrame(stability)
    stability.to_csv(out/'cluster_stability.csv',index=False)
    profiles=[]; examples=[]
    for cluster in range(k):
        idx=np.flatnonzero(km.labels_==cluster)
        group=train.iloc[idx]
        genres=[g for s in group.genres for g in json.loads(s)]
        top_genres=pd.Series(genres).value_counts().head(3).index.tolist()
        profiles.append({'cluster':cluster,'train_rows':len(group),
                         'median_year':group.premiere_year.median(),
                         'median_runtime':(group.runtime_minutes.dropna().median()
                                           if group.runtime_minutes.notna().any() else None),
                         'missing_runtime_share':group.runtime_minutes.isna().mean(),
                         'top_genres':', '.join(top_genres),
                         'top_country':group.broadcast_country.mode().iloc[0],
                         'streaming_share':group.distribution.eq('Streaming').mean(),
                         'high_rated_share_descriptive':group.high_rated.mean()})
        distances=np.linalg.norm(P[idx]-km.cluster_centers_[cluster],axis=1)
        for j in np.argsort(distances)[:3]:
            row=group.iloc[j]
            examples.append({'cluster':cluster,'show_id':int(row.show_id),'name':row['name'],
                             'year':row.premiere_year,'runtime':row.runtime_minutes,
                             'genres':row.genres,'network':row.network,'source_url':row.source_url})
    profiles=pd.DataFrame(profiles); examples=pd.DataFrame(examples)
    profiles.to_csv(out/'cluster_profiles.csv',index=False)
    examples.to_csv(out/'cluster_examples.csv',index=False)
    coords=train[['show_id','name','high_rated']].copy()
    coords['PC1']=P[:,0];coords['PC2']=P[:,1];coords['cluster']=km.labels_
    coords.to_csv(out/'pca_coordinates.csv',index=False)
    summary={'encoded_features':Z.shape[1],'components_90':pca.n_components_,
             'variance_2d':float(pca.explained_variance_ratio_[:2].sum()),
             'variance_retained':float(pca.explained_variance_ratio_.sum()),'selected_k':k,
             'silhouette':float(quality.loc[quality.k.eq(k),'silhouette'].iloc[0]),
             'mean_seed_ari':float(stability.adjusted_rand_index.mean()),
             'stability_scope':'initialisation seeds on the same rows, not population stability',
             'scaling':'Numeric fields and their missing indicators standardised; one-hot and genre flags stay 0/1.'}
    return prep,pca,km,variance,quality,stability,profiles,examples,coords,summary
