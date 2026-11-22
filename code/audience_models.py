"""Models for catalogue data and the separate audience-reception experiments."""
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from audience_features import ALL_EPISODES, IMDB, IMDB_ALL, REACTION, REVIEWS, SEASON, STRUCTURE
from final_models import metadata

LENS = ['lens_votes','lens_mean','lens_std','lens_relative_mean','lens_prior_count_mean','lens_profile_share']
EXPERIMENTS = {
    'catalogue': [],
    'production': ['structure','credits','episode_text'],
    'episode_ratings_only': ['episode_ratings','season_ratings'],
    'first_three': ['structure','credits','episode_text','episode_ratings'],
    'first_season': ['structure','credits','episode_text','episode_ratings','season_ratings'],
    'all_episodes': ['structure','credits','episode_text','episode_ratings','season_ratings','all_episode_ratings'],
    'with_reviews': ['structure','credits','episode_text','episode_ratings','season_ratings','review'],
    'with_lens_ratings': ['structure','credits','episode_text','episode_ratings','season_ratings','review','lens_ratings'],
    'with_rater_history': ['structure','credits','episode_text','episode_ratings','season_ratings','review','lens'],
    'imdb_show_only_local': ['imdb_show'],
    'imdb_only_local': ['imdb'],
    'cross_site_local': ['structure','credits','episode_text','episode_ratings','season_ratings','review','imdb'],
    'all_sources_local': ['structure','credits','episode_text','episode_ratings','season_ratings','all_episode_ratings',
                          'review','lens','imdb','imdb_all'],
}
PARAMETERS = {
    'logistic_c0.01': ('logistic', .01),
    'logistic_c0.05': ('logistic', .05),
    'logistic_c0.25': ('logistic', .25),
    'ridge_alpha10': ('ridge', 10),
    'ridge_alpha100': ('ridge', 100),
    'hist_leaves7': ('hist_classifier', 7),
    'hist_leaves15': ('hist_classifier', 15),
    'hist_reg_leaves7': ('hist_regressor', 7),
    'hist_reg_leaves15': ('hist_regressor', 15),
    'blend_lr25': ('blend', .25),
    'blend_lr40': ('blend', .40),
    'blend_lr60': ('blend', .60),
    'blend_lr75': ('blend', .75),
}
BLEND_LR, BLEND_HIST = 'logistic_c0.05', 'hist_leaves7'


class CreditHistory:
    """Smoothed outcomes of other training shows sharing performers or staff."""

    def fit(self, credits, ratings):
        self.prior = float(np.mean(ratings))
        totals = defaultdict(float)
        counts = defaultdict(int)
        for people, rating in zip(credits, ratings):
            for person in people:
                totals[person] += rating
                counts[person] += 1
        self.people = {person: (counts[person], (total + 5*self.prior)/(counts[person]+5))
                       for person, total in totals.items()}
        return self

    def transform(self, credits):
        features = []
        for people in credits:
            row = []
            for is_cast in (True, False):
                credits_group = [person for person in people if person.startswith('cast:') == is_cast]
                known = [self.people[person] for person in credits_group if person in self.people]
                scores = [value for _, value in known]
                row.extend([len(known)/max(len(credits_group),1),
                            np.mean(scores) if scores else self.prior,
                            max(scores,default=self.prior), min(scores,default=self.prior)])
            features.append(row)
        return np.asarray(features)

    def fit_transform(self, frame):
        output = np.empty((len(frame), 8))
        cv = StratifiedKFold(3, shuffle=True, random_state=314)
        for fit, validation in cv.split(frame, frame.high_rated):
            history = CreditHistory().fit(frame.credits.iloc[fit], frame.rating.iloc[fit])
            output[validation] = history.transform(frame.credits.iloc[validation])
        self.fit(frame.credits, frame.rating)
        return output


class AudienceEncoder:
    def __init__(self, experiment):
        self.blocks = EXPERIMENTS[experiment]
        self.use_metadata = experiment not in ('episode_ratings_only','imdb_show_only_local','imdb_only_local')

    def numeric(self, frame):
        columns = []
        for name, fields in [('structure',STRUCTURE),('episode_ratings',REACTION),('season_ratings',SEASON),
                             ('all_episode_ratings',ALL_EPISODES),
                             ('review',REVIEWS),('lens_ratings',LENS[:3]),('lens',LENS),
                             ('imdb_show',IMDB[:2]),('imdb',IMDB),('imdb_all',IMDB_ALL)]:
            if name in self.blocks:
                columns.extend(fields)
        return frame[columns].to_numpy(dtype=float)

    def embeddings(self, vectors):
        if not self.use_metadata:
            return []
        blocks = [vectors['plot_summary']*3]
        if 'episode_text' in self.blocks:
            blocks.append(vectors['episode_text']*3)
        if 'review' in self.blocks:
            blocks.append(vectors['review_text']*3)
        return blocks

    def fit_transform(self, frame, vectors):
        fields = self.numeric(frame)
        if self.use_metadata:
            self.metadata = metadata(scale=True)
            basic = self.metadata.fit_transform(frame)
        else:
            basic = np.zeros((len(frame),0))
        if 'credits' in self.blocks:
            self.credits = CreditHistory()
            fields = np.column_stack([fields,self.credits.fit_transform(frame)])
        if fields.shape[1]:
            self.imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
            self.scaler = StandardScaler()
            linear_fields = self.scaler.fit_transform(self.imputer.fit_transform(fields))
        else:
            linear_fields = fields
        return (np.column_stack([basic,linear_fields,*self.embeddings(vectors)]),
                np.column_stack([basic,fields]))

    def transform(self, frame, vectors):
        fields = self.numeric(frame)
        basic = self.metadata.transform(frame) if self.use_metadata else np.zeros((len(frame),0))
        if 'credits' in self.blocks:
            fields = np.column_stack([fields,self.credits.transform(frame.credits)])
        linear_fields = self.scaler.transform(self.imputer.transform(fields)) if fields.shape[1] else fields
        return (np.column_stack([basic,linear_fields,*self.embeddings(vectors)]),
                np.column_stack([basic,fields]))


def estimator(name):
    family, value = PARAMETERS[name]
    if family == 'logistic':
        return LogisticRegression(C=value, solver='liblinear', class_weight='balanced', random_state=42), 0, False
    if family == 'ridge':
        return Ridge(alpha=value), 0, True
    options = {'max_leaf_nodes':value, 'learning_rate':.07, 'max_iter':110,
               'min_samples_leaf':28, 'l2_regularization':2., 'early_stopping':False, 'random_state':42}
    if family == 'hist_regressor':
        return HistGradientBoostingRegressor(**options), 1, True
    return HistGradientBoostingClassifier(class_weight='balanced', **options), 1, False


def fit_candidate(name, matrices, rows):
    if PARAMETERS[name][0] == 'blend':
        return (fit_candidate(BLEND_LR,matrices,rows),fit_candidate(BLEND_HIST,matrices,rows))
    model, position, regression = estimator(name)
    label = rows.rating if regression else rows.high_rated
    model.fit(matrices[position], label)
    return model


def score_candidate(name, model, matrices):
    if PARAMETERS[name][0] == 'blend':
        weight = PARAMETERS[name][1]
        return weight*score_candidate(BLEND_LR,model[0],matrices)+(1-weight)*score_candidate(BLEND_HIST,model[1],matrices)
    _, position, regression = estimator(name)
    return model.predict(matrices[position]) if regression else model.predict_proba(matrices[position])[:,1]
