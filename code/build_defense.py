"""Write the nine-slide speech and practical demo plan from measured findings."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'results/final'
    summary = json.loads((out / 'analysis_summary.json').read_text())
    findings = json.loads((out / 'reviewed_findings.json').read_text())
    models = {row['family']: row for row in summary['models']}
    delta = next(row for row in summary['deltas']
                 if row['baseline'] == 'engineered_metadata' and row['new'] == 'minilm_combined')
    mlp = summary['mlp']
    primary = models[summary['primary']]['name'].replace(', full grid', '')
    slides = [
        ('Bekzat', 'The question',
         'Our project asks whether a plot synopsis helps identify TV series with a TVmaze rating of eight or higher, '
         'when structured metadata are already available. We started with metadata at the Midterm and added '
         'ensemble and neural models at the Endterm. For the Final, we test the story-text hypothesis. '
         'We compare metadata alone, text alone and both sources together. A positive result is useful, '
         'but a negative result also answers the question. Our goal is to measure the added information '
         'fairly and explain the remaining mistakes. One prediction corresponds to one series in the catalogue.'),
        ('Bekzat', 'The same objects and evaluation',
         'The original snapshot contains fourteen thousand and fifty-five series. After filtering to rated English '
         'scripted series, we have two thousand three hundred and thirty-nine examples. Four hundred and four '
         'have ratings of at least eight. We preserve the original split: eighteen seventy-one training '
         'series and four hundred and sixty-eight test series. Five outer training folds evaluate the models. '
         'Inner folds choose settings. Positive-class F1 is our main metric, because predicting below eight '
         'for everything gives about eighty-three percent accuracy while finding no highly rated series.'),
        ('Bekzat', 'Feature engineering and complete grids',
         'We added genre count, genre combinations, premiere decade and runtime buckets. Missing-value filling '
         'and category encoding learn only from the corresponding fitting partition. We also tested cross-fitted '
         'target encoding for sparse categories. Following the feedback, each ensemble has a complete grid of '
         'two hundred and fifty-six configurations. The neural grid has sixty-four, covering four architectures, '
         'four regularisation values and four learning rates. The report shows every range and whether the selected '
         'value is inside the range or at its boundary. The rating itself is never a prediction input.'),
        ('Gapar', 'Two text representations',
         'We use two main approaches to represent the synopsis. TF-IDF gives numerical weights to words and '
         'word pairs, and logistic regression learns the classification task. The vocabulary is learned within '
         'each training fold. MiniLM is a frozen pretrained transformer with six layers. It converts a synopsis '
         'into three hundred and eighty-four numbers. We then train a logistic classifier on those vectors. '
         'The encoder weights remain unchanged. Both approaches use the same series and labels. Combined '
         'models receive metadata and text, with regularisation and text weight chosen on training folds. '
         'The trained classifiers and preprocessing are saved together.'),
        ('Gapar', 'Measured results',
         f'The controlled comparison uses the same metadata logistic pipeline. Its outer CV F1 is '
         f'{models["engineered_metadata"]["cv_f1_mean"]:.3f}. Adding MiniLM gives '
         f'{models["minilm_combined"]["cv_f1_mean"]:.3f}. On the reused test benchmark, F1 changes from '
         f'{models["engineered_metadata"]["test_f1"]:.3f} to {models["minilm_combined"]["test_f1"]:.3f}. '
         f'The paired bootstrap interval for this difference is {delta["low_95"]:.3f} to {delta["high_95"]:.3f}. '
         'The interval and fold variation matter more than a single higher number. The table also compares '
         f'the stronger metadata reference. Training cross-validation selects {primary} as the primary model. '
         'We do not select the winner by its test score.'),
        ('Gapar', 'Learning curves and checkpoint',
         f'The chosen neural architecture has sixty-four and thirty-two hidden units. Its validation loss reaches '
         f'the minimum at epoch {mlp["best_epoch"]}. Training loss keeps decreasing after that point, but '
         f'validation loss increases. The gap grows from {mlp["gap_at_best"]:.3f} to {mlp["gap_at_stop"]:.3f}. '
         'This indicates overfitting: improvements on fitting examples stop transferring to validation examples. '
         f'The run stops at epoch {mlp["stopped_epoch"]} after patience, then restores the actual minimum-loss '
         'weights. The saved model also keeps preprocessing fitted on its fitting subset. We verified that '
         'reloading it reproduces the selected validation loss.'),
        ('Aibek', 'Specific errors', findings['defense_cases']),
        ('Aibek', 'Saved-model application',
         'The local app accepts metadata and an English synopsis for a new series. It compares metadata, text '
         'and combined predictions. The browser sends a row to the Python server, which applies saved '
         'preprocessing and classifiers. Pressing Predict does not train anything. This fictional example '
         'has no known rating, so it demonstrates inference rather than accuracy. Empty text uses the '
         'metadata fallback. Blank numbers use training medians; invalid values produce a clear input error.'),
        ('Aibek', 'Conclusion and limitations',
         findings['defense_conclusion']+' '
         'The evidence has limits. We reused the course test benchmark, and related franchises can appear '
         'in both random partitions. We do not have vote counts, and pretrained-data overlap is unknown. '
         'A synopsis describes the premise more directly than acting or execution quality. A stronger next '
         'experiment would use a new temporal or franchise-grouped evaluation. The submitted code, '
         'models and notebook reproduce the result.'),
    ]
    text = ('# Final defense: seven-minute talk and live demonstration\n\n'
            'Slides 1–3: **Bekzat**. Slides 4–6: **Gapar**. Slides 7–9: **Aibek**. '
            'Aim for about 2 minutes 20 seconds each. Learn the meaning and speak to the audience; '
            'the text is a rehearsal aid. The app demonstration follows the seven-minute slides.\n\n')
    for index, (speaker, title, speech) in enumerate(slides, 1):
        text += f'## Slide {index} — {speaker}: {title}\n\n{speech}\n\n'
    counts = {}
    for speaker in ['Bekzat', 'Gapar', 'Aibek']:
        counts[speaker] = sum(len(re.findall(r"\b[\w'-]+\b", speech))
                              for owner, _, speech in slides if owner == speaker)
    text += '## Timing\n\n'
    text += '\n'.join(f'- {speaker}: approximately {count} words.' for speaker, count in counts.items())
    text += '''

Rehearse with a timer. At roughly 115–125 words per minute, the talk is close to seven minutes. Reduce pauses or trim a sentence if needed; keep the numeric findings, error explanations and conclusion.

## Live app demonstration — Bekzat, about 2.5 minutes

Before the defense: start `python app.py`, open `http://127.0.0.1:8765`, and keep the notebook's saved-model inference cell ready. Use saved models; do not start the full grid.

1. **0:00–0:35 — Real snapshot example.** Choose Person of Interest, press Run prediction and show the three outputs and observed rating reference. Say: “The rating is shown for comparison. It is excluded from the feature row.” Explain the actual error or success using the report.
2. **0:35–1:20 — New input.** Select Your own series. Enter year 2026, runtime 45, Netflix, US, Streaming, Drama and Mystery. Enter: “An astronomer investigates a signal with a retired engineer.” Press Run prediction. Say: “These are outputs from saved models on a new input. We do not know its true rating.”
3. **1:20–1:50 — Missing values.** Clear the synopsis and runtime, predict again. Say: “The metadata model is available. Runtime uses the training median. Text models need a synopsis.”
4. **1:50–2:15 — Invalid input.** Enter runtime -5 and press Predict. The browser prevents the invalid value; the server also rejects it. Explain that input validation protects the prediction path. Restore 45.
5. **2:15–2:30 — Saved model.** Show notebook section 7: `joblib.load(...)` followed by `.predict(...)`. Say: “The learned preprocessing and classifier are loaded together. This call performs inference without fitting again.”

The Final brief allows about ten minutes for slides and app, followed by five minutes of questions. The teacher’s personal feedback asks for a seven-minute talk, so keep the slides to seven minutes and use the remaining presentation time for the app.

## Questions everyone should be able to answer

- **What is the target?** A binary indicator that the observed TVmaze rating is at least 8. We do not predict the exact numeric rating.
- **Why F1?** Only 17.3% are positive. Majority-class accuracy hides complete failure to recover positives.
- **What did the text add?** Answer using the controlled metadata+LR versus metadata+MiniLM+LR comparison, its fold variation and paired interval. The strongest metadata family is a separate reference.
- **What is learned in MiniLM?** Its encoder is pretrained and frozen. Our logistic classifier learns the rating task from its vectors.
- **Where is deep learning?** The six-layer MiniLM transformer supplies text features. We also compare four trained MLP architectures on metadata.
- **How is leakage controlled?** Explicit input whitelist, fold-fitted preprocessing, cross-fitted target encoding, saved disjoint IDs and training-CV selection. The reused benchmark remains an acknowledged limitation.
- **Why restore an earlier epoch?** Validation loss worsens after its minimum while fitting loss improves, indicating overfitting. The saved checkpoint retains the minimum-loss weights.
- **What happens when you press Predict?** Input validation, feature-row construction, saved preprocessing, saved classifier, JSON response and display. No training runs.
- **Is a score of 0.7 a reliable 70% chance?** No. Class weighting and the lack of calibration mean these are model scores. The class threshold is 0.5.
- **Why are the metrics modest?** The target is subjective; vote counts and execution-quality measures are missing. A premise or channel only partially indicates audience response. We report the actual limitations and do not change the test to raise the score.
- **What next?** New temporal or franchise-grouped test data, rating reliability if lawful vote-count information becomes available, and threshold selection performed inside training folds if required by a defined use case.
'''
    (ROOT / 'docs/DEFENSE_EN.md').write_text(text, encoding='utf-8')
    (out / 'speaker_notes.json').write_text(json.dumps([
        {'slide': index, 'speaker': speaker, 'title': title, 'speech': speech}
        for index, (speaker, title, speech) in enumerate(slides, 1)], indent=2), encoding='utf-8')
    print('Speech words by speaker:', counts)


if __name__ == '__main__':
    main()
