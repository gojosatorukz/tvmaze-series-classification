# TVmaze rating classifiers

## Task and intended use

Exploratory classification of a TVmaze catalogue series as having an observed aggregate rating of at least 8/10. Intended for reproducible ML experimentation and local demonstration. The output is a binary class plus an uncalibrated model score. It does not estimate an exact star rating or establish the quality of a future production.

## Training data

TVmaze public API snapshot, 6 October 2026; 60 raw pages, 14,055 original objects. The labelled cohort contains 2,339 rated English scripted series; 404 positives. The saved training split contains 1,871 series and the reused test benchmark contains 468. All retained objects have plot descriptions. Raw responses and hashes, cleaning and saved object IDs accompany the experiment. Data are CC BY-SA 4.0 with TVmaze attribution.

## Inputs and models

Metadata: premiere year, episode runtime, TV network, web channel, broadcaster/platform country, distribution and genres. Derived features use only these inputs. Text variants additionally use an English synopsis. Rating, label, show ID, title, URLs, status and popularity are excluded from the prediction input whitelist.

Models include logistic classifiers with metadata, TF-IDF, LSA or frozen MiniLM vectors; cross-fitted target encoding; tuned Random Forest, Gradient Boosting and MLP; and earlier-stage references. The app manifest records the training-CV-selected metadata, text and combined models. Complete metrics, search settings, error examples and the selection record are in `results/final/` after extracting the artefact archive.

The pretrained encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, using quantised local ONNX weights. It produces 384-dimensional normalised vectors and truncates to 256 tokens. The encoder is frozen; logistic regression learns the rating task. Apache-2.0 applies to those model files.

## Evaluation

Positive-class F1 is primary. Five saved outer folds evaluate each tuned procedure with three-fold inner exhaustive searches. Full-training five-fold searches choose final settings. Preprocessing is fitted inside the relevant fitting partitions. The Final test predictions use the same saved objects as the earlier course stages; this is not a new independent test.

Paired stratified percentile bootstrap intervals resample fixed test objects 2,000 times with seed 42. They condition on fitted models and do not include retraining or franchise-dependence uncertainty. Multiple representation comparisons are exploratory. Model and threshold selection must not be made from the largest observed test score.

## Known limits

The cohort is selected by early API index, language, series type and rating availability. It is not representative of all television or unreleased productions. Ratings are subjective and vote counts are unavailable. Related series may appear across random partitions. Current metadata are catalogue snapshots and need not match premiere-time information. Pretraining overlap is unknown. Long synopses can be truncated. A synopsis describes a premise more directly than execution quality or audience preferences.

Unknown categories follow the saved encoder policy; blank numeric inputs use training medians. Missing text triggers the metadata-only path. Inputs outside the training population can produce unreliable outputs. Scores are not calibrated probabilities. A fresh temporal or franchise-grouped evaluation is required before making stronger generalisation claims.

## Reproduction and ownership

`START_FINAL.md` and the public README explain dependency installation, artefact extraction, verification, full retraining and app launch. The model weights and preprocessing are versioned together. Verify the supplied files before use and only unpickle trusted artefacts.

Bekzat Rashiduly owns the technical development. Course presentation participants are documented separately in the report; presentation participation is not a claim of technical implementation. The formal course report includes the required description of assistance and tools.
