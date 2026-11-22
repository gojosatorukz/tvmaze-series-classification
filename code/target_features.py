"""Cross-fitted target encoding for sparse broadcaster and genre categories."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from final_models import DerivedMetadata
from final_text import MiniLMFeatures
from project import CATEGORICAL, FEATURES, NUMERIC, SEED, GenreEncoder


def encoded_metadata():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([
        ('target', TargetEncoder(target_type='binary', smooth=10, cv=5,
                                 shuffle=True, random_state=SEED)),
        ('scale', StandardScaler()),
    ])
    prepare = ColumnTransformer([
        ('numeric', numeric, NUMERIC + ['genre_count']),
        ('categories', categorical, CATEGORICAL + ['genre_combination', 'premiere_decade', 'runtime_bucket']),
        ('genres', GenreEncoder(), ['genres']),
    ])
    return Pipeline([('derive', DerivedMetadata()), ('encode', prepare)])


def target_grids():
    definitions = {}
    for name, combined in [('target_encoded_metadata', False), ('target_minilm_combined', True)]:
        blocks = [('metadata', encoded_metadata(), FEATURES)]
        if combined:
            blocks.append(('text', MiniLMFeatures(), 'plot_summary'))
        model = Pipeline([
            ('prepare', ColumnTransformer(blocks)),
            ('model', LogisticRegression(class_weight='balanced', solver='liblinear',
                                         max_iter=3000, random_state=SEED)),
        ])
        grid = {'model__C': [.001, .01, .1, 1],
                'prepare__metadata__encode__categories__target__smooth': [5, 10, 20, 50]}
        if combined:
            grid['prepare__transformer_weights'] = [
                {'metadata': 1, 'text': value} for value in [.3, 1, 3, 6]]
        definitions[name] = model, grid
    return definitions
