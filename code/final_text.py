"""Frozen MiniLM text features, loaded locally through ONNX Runtime."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from final_models import metadata
from project import FEATURES, SEED

ROOT = Path(__file__).resolve().parents[1]
_encoder = None
_lock = threading.RLock()


class LocalEncoder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        ort.disable_telemetry_events()
        folder = ROOT / 'assets/minilm'
        manifest = json.loads((folder / 'manifest.json').read_text())
        for filename in ['tokenizer.json', 'onnx/model_qint8_arm64.onnx']:
            expected = next(item['sha256'] for item in manifest['files'] if item['name'] == filename)
            if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != expected:
                raise ValueError(f'MiniLM file integrity check failed: {filename}')
        self.tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / 'onnx/model_qint8_arm64.onnx'),
                                            sess_options=options, providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.cache = OrderedDict()

    def encode(self, values):
        texts = [str(text) for text in values]
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        # Single-text batches keep dynamic quantisation independent of other inputs.
        for start in range(len(missing)):
            batch = missing[start:start+1]
            encodings = self.tokenizer.encode_batch(batch)
            values_by_name = {
                'input_ids': np.asarray([item.ids for item in encodings], dtype=np.int64),
                'attention_mask': np.asarray([item.attention_mask for item in encodings], dtype=np.int64),
                'token_type_ids': np.asarray([item.type_ids for item in encodings], dtype=np.int64),
            }
            output = self.session.run(None, {name: values_by_name[name] for name in self.inputs})[0]
            if output.ndim == 3:
                mask = values_by_name['attention_mask'][..., None]
                output = (output*mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            output = output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
            for text, vector in zip(batch, output):
                self.cache[text] = vector.astype(np.float32)
        vectors = np.stack([self.cache[text] for text in texts])
        while len(self.cache) > 4096:
            self.cache.popitem(last=False)
        return vectors


def encode_texts(texts):
    global _encoder
    with _lock:
        if _encoder is None:
            _encoder = LocalEncoder()
        return _encoder.encode(texts)


class MiniLMFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        return encode_texts(np.asarray(X).ravel())

    def get_feature_names_out(self, input_features=None):
        return np.array([f'minilm_{index}' for index in range(384)], dtype=object)


def semantic_grids():
    definitions = {}
    for name, combined in [('minilm_text', False), ('minilm_combined', True)]:
        blocks = []
        if combined:
            blocks.append(('metadata', metadata(scale=True), FEATURES))
        blocks.append(('text', MiniLMFeatures(), 'plot_summary'))
        model = Pipeline([
            ('prepare', ColumnTransformer(blocks)),
            ('model', LogisticRegression(class_weight='balanced', solver='liblinear',
                                         max_iter=3000, random_state=SEED)),
        ])
        space = {'model__C': [.001, .01, .1, 1]}
        if combined:
            space['prepare__transformer_weights'] = [
                {'metadata': 1, 'text': weight} for weight in [.3, 1, 3, 6]]
        definitions[name] = model, space
    return definitions
