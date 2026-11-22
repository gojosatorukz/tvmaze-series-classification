"""Render the reception extension write-up and F1 comparison from saved results."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/audience'
os.environ['MPLCONFIGDIR'] = str(ROOT/'.runtime/audience-matplotlib')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


NAMES = {
    'final_gradient_boosting':'Original Final gradient boosting',
    'catalogue':'Metadata + synopsis, tuned threshold',
    'production':'Catalogue + episode text + production',
    'episode_ratings_only':'TVmaze episode ratings only',
    'first_three':'Production + first three episode ratings',
    'first_season':'Add first season episode ratings',
    'all_episodes':'Add all aired TVmaze episode ratings',
    'with_reviews':'Add editorial review text',
    'with_lens_ratings':'Add MovieLens rating aggregates',
    'with_rater_history':'Add earlier rater behaviour',
    'imdb_show_only_local':'IMDb series rating and vote count only',
    'imdb_only_local':'IMDb series and early episode ratings',
    'cross_site_local':'TVmaze, editorial reviews and IMDb',
    'all_sources_local':'All sources including rater history',
}


def read(path):
    return json.loads(path.read_text())


def plot_results(selection, scores, transfer, path):
    labels = list(scores)
    fig, ax = plt.subplots(figsize=(10,7.2),layout='constrained')
    y = np.arange(len(labels))
    width = .23
    for offset, name, color, data in [
        (-width, 'Outer CV mean','#7591A5',{key:value['mean'] for key,value in selection['outer_cv'].items()}),
        (0, 'Frozen 468-show benchmark','#13586C',{key:value['f1'] for key,value in scores.items()}),
        (width, 'Separate TVmaze catalogue slice','#D98343',{key:value['f1'] for key,value in transfer.items()}),
    ]:
        indices = [i for i,key in enumerate(labels) if key in data]
        ax.barh(y[indices]+offset,[data[labels[i]] for i in indices],height=width-.025,label=name,color=color)
    ax.axvline(.8,color='#5D5D5D',linestyle='--',linewidth=1.2)
    ax.text(.805,.985,'F1 goal 0.80',transform=ax.get_xaxis_transform(),fontsize=9,ha='left',va='top',color='#4C4C4C')
    ax.set(yticks=y,yticklabels=[NAMES[key] for key in labels],xlim=(0,1),xlabel='Positive-class F1')
    ax.invert_yaxis()
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=3,frameon=False,fontsize=8)
    fig.savefig(path,dpi=160)
    plt.close(fig)


def main():
    selection = read(OUT/'training_selection.json')
    scores = {row['experiment']:row for row in read(OUT/'benchmark_metrics.json')}
    transfer_data = read(OUT/'transfer_metrics.json')
    transfer = {row['experiment']:row for row in transfer_data['metrics']}
    primary, local = selection['public_winner'], selection['local_winner']
    base, primary_score, local_score = (scores[key]['f1'] for key in ('final_gradient_boosting',primary,local))
    observed = max(scores,key=lambda name:scores[name]['f1'])
    observed_score = scores[observed]['f1']
    original = pd.read_csv(ROOT/'results/final/model_comparison.csv').set_index('family').loc['gradient_boosting']
    ci = scores[primary]['paired_vs_final']
    audit = read(ROOT/'data/audience/source_audit.json')
    lens = read(ROOT/'data/audience/movielens_audit.json')
    coverage = selection['coverage']
    subset = pd.read_csv(OUT/'coverage_metrics.csv')
    matched = subset[subset.group.eq('has_movielens_history')]
    matched_n = int(matched.n.iloc[0]) if len(matched) else 0
    matched_pos = int(matched.positives.iloc[0]) if len(matched) else 0
    ablations = pd.read_csv(OUT/'paired_ablation.csv').set_index('additional_input')
    examples = read(OUT/'audience_error_examples.json')
    example_lines = []
    for row in examples:
        episode_mean = f'{row["all_episode_mean"]:.2f}' if row['all_episode_mean'] is not None else 'unavailable'
        example_lines.append(f'- **{row["name"]}** ({row["kind"]}). Observed show rating {row["true_rating"]:.1f}. '
                             f'Mean observed episode rating {episode_mean} over {row["rated_episodes"]} rated episodes. '
                             f'The new model score moves from {row["new_score"]:.3f} to '
                             f'{row["score_with_episode_ratings_masked"]:.3f} when its episode ratings are masked. '
                             f'The decision threshold is {selection["selection"][primary]["threshold"]:.3f}.')
    chart = OUT/'audience_comparison.png'
    plot_results(selection,scores,transfer,chart)
    table = ['| Input configuration | Outer CV F1 mean ± SD | Test F1 | Test precision | Test recall | Separate slice F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, item in scores.items():
        cv = ({'mean':original.cv_f1_mean,'std':original.cv_f1_sd} if key=='final_gradient_boosting'
              else selection['outer_cv'][key])
        value = f'{transfer[key]["f1"]:.3f}' if key in transfer else '—'
        table.append(f'| {NAMES[key]} | {cv["mean"]:.3f} ± {cv["std"]:.3f} | {item["f1"]:.3f} | {item["precision"]:.3f} | {item["recall"]:.3f} | {value} |')
    reached = observed_score >= .8
    target = 'At least one test result reached the 0.80 F1 target.' if reached else 'Every result stayed below the 0.80 F1 target on the full frozen benchmark.'
    text = f'''# Can audience information improve TVmaze rating classification?

**Bekzat Rashiduly · Follow-up to the TVmaze course experiment**

The original Final compared metadata and synopses on 2,339 series. This follow-up keeps the target **TVmaze rating ≥ 8/10**, the original show IDs and folds, then adds production data, episode plots and audience information. It asks how performance changes once those sources are available for a released series.

The public-input model selected by training cross-validation was **{NAMES[primary]}**. Its F1 on the same 468 series changed from **{base:.3f} to {primary_score:.3f}**. The paired stratified bootstrap interval for the difference was **[{ci['low_95']:+.3f}, {ci['high_95']:+.3f}]**. The training-selected local IMDb experiment reached **{local_score:.3f}**. The highest observed benchmark score was **{observed_score:.3f}** for **{NAMES[observed]}**. This is a descriptive maximum across tried variants. The training-selected model was already frozen. {target}

## What was added

- **TVmaze episodes and credits.** Exact `show_id` join. {coverage['matched_details']:,} details records, {coverage['has_episode_text']:,} series with text from the first three episodes, {coverage['has_first_three_ratings']:,} with ratings on those episodes and {coverage['has_season_one_ratings']:,} with at least one rated episode in season one. Text uses frozen MiniLM. The original TVmaze show rating remains the target. Episode scores come from separate episode voting. Cast and crew features summarize outcomes of other shows in the fitting partition, with three-fold cross-fitting to build training features.
- **Editorial review text.** {audit['celluloid']['usable_series']} Celluloid matches by unique normalized title and premiere year, with the linked article, byline and a last-updated date through 6 October 2026. The text block excludes score-bearing sentences and the separate verdict sections. This is sparse coverage from one byline.
- **Rater history.** MovieTweetings matches {audit['movietweetings']['matched_series']} series. MovieLens 32M covers {lens['matched_series']} series with {lens['matched_events']:,} events from {lens['matched_raters']:,} pseudonymous raters. For {lens['historical_profiles']:,} of those ratings, at least ten strictly earlier ratings give a personal reference mean. A relative rating subtracts that mean from the current value. The model uses only per-series aggregates, with no individual user IDs. The benchmark contains {matched_n} MovieLens-covered titles ({matched_pos} positive).
- **Local IMDb comparison.** Exact IMDb ID joins cover {audit['imdb_local']['matched_series']:,} series, including {audit['imdb_local']['first_three_episode_matched_series']:,} with ratings on at least one of the first three episodes. Aggregate rating and vote count come from the official downloadable datasets. Information courtesy of IMDb ([IMDb](https://www.imdb.com)). Used for individual non-commercial analysis under its terms. Title-level IMDb files and fitted IMDb-dependent model binaries stay local.

## Evaluation

Each candidate and its F1 decision threshold were selected from inner validation predictions. The five outer folds use saved training IDs. A final five-fold search freezes the candidate and threshold before reading the 468-series results. The original Final gradient boosting is included at its originally reported threshold, while the catalogue row controls for adding threshold tuning and the new model search. Read an input block's change against its nearest control. The exact fold and threshold tables are included.

{chr(10).join(table)}

The public model changes precision from **{scores['final_gradient_boosting']['precision']:.3f} to {scores[primary]['precision']:.3f}** and recall from **{scores['final_gradient_boosting']['recall']:.3f} to {scores[primary]['recall']:.3f}**. It recovers fewer positives and produces a much shorter list of false positives. Its confusion counts on the 468 shows are {scores[primary]['tn_fp_fn_tp'][0]} true negatives, {scores[primary]['tn_fp_fn_tp'][1]} false positives, {scores[primary]['tn_fp_fn_tp'][2]} false negatives and {scores[primary]['tn_fp_fn_tp'][3]} true positives.

The episode-only and IMDb-only rows measure two sources of audience signal separately. Against its direct benchmark control, the first-three episode block changes F1 by **{ablations.loc['first_three','f1_difference']:+.3f}**. Editorial review text changes it by **{ablations.loc['with_reviews','f1_difference']:+.3f}**, and relative reviewer behaviour changes it by **{ablations.loc['with_rater_history','f1_difference']:+.3f}** after the ordinary MovieLens rating aggregates. Each row reselects its method and decision threshold from training folds. These differences measure pipelines, with uncertainty and sparse source coverage discussed in `paired_ablation.csv` and `coverage_metrics.csv`.

The additional catalogue slice consists of **{transfer_data['rows']} different TVmaze IDs ({transfer_data['positives']} positives)** from fixed index pages 60–99. Exact IMDb duplicates and title-year overlaps with the old index were removed. The previously frozen models were applied without recalibration. The public model reached **{transfer[primary]['f1']:.3f}** there, versus **{transfer['final_gradient_boosting']['f1']:.3f}** for the original Final model. The local IMDb-assisted model reached **{transfer[local]['f1']:.3f}**.

The two cohorts have different positive shares, {81/468:.1%} in the frozen benchmark and {transfer_data['positives']/transfer_data['rows']:.1%} in the separate slice. The public model's benchmark gain did not carry over to the other catalogue. Differences in release mix, rating participation and class prevalence matter when applying a fixed threshold. This is why the extension reports both evaluations.

## Examples from the unchanged test set

{chr(10).join(example_lines)}

These examples display both repairs and newly introduced errors. Episode ratings are observations about episodes. The model and audience can disagree on the overall show score even when episodes rate highly.

## Scope of the result

Episode and IMDb votes were retrieved in November. The course target is an October snapshot. The sources can include votes cast after that date, and the public API does not supply their historical per-episode vote timestamps. These numbers measure retrospective catalogue classification using audience information. The second slice checks transfer to other catalogue entries using current ratings. A genuine forecast after the first three episodes would require archived input scores at a declared cutoff and show ratings from a later date. Catalogued cast/crew and season structure can also change after premiere.

The original synopsis hypothesis remains a separate content-only comparison. New audience signals can be informative without supporting a conclusion about synopsis quality. Coverage by platform and show popularity can bias audience inputs. The reused 468-row test, related franchises, and limited review coverage bound generalisation claims.

![F1 on the frozen benchmark and the disjoint catalogue slice](results/audience/audience_comparison.png)

## Reproduce and demonstrate

From the extracted project folder with the Final Python environment active:

```bash
python code/predict_audience.py --show-id 2
python code/verify_audience.py
python code/audience_experiment.py --lens data/audience/movielens_aggregate.csv --out results/audience_public_reproduction
```

The final command reruns the public variants. The complete audience run also takes `--local-imdb` pointing to the file prepared locally from IMDb. `training_selection.json`, `outer_cv.csv`, `tuning.csv`, `coverage_metrics.csv` and `transfer_metrics.json` contain the evidence. Licensed source snapshots and aggregates are in `data/audience`.

For the oral defence: “The earlier experiment tested information inside the synopsis. Here I tested evidence from audiences after release. I used the same test IDs and tuned the threshold only inside training folds. I then applied the frozen model to a second set of shows. The audience-assisted score describes a different information setting from predicting a new series before anyone has watched it.”

## Sources and licences

- [TVmaze API](https://www.tvmaze.com/api). TVmaze data and derived episode/credit projections, CC BY-SA. [Separate episode voting](https://www.tvmaze.com/blogs/30/voting-on-episodes-from-episode-list-now-possible).
- [Celluloid review data](https://celluloidscore.com/about/data/), CC BY 4.0. Review excerpts were matched by title and year, trimmed and screened. Title links, article links, update dates and author attribution accompany the retained records.
- [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) under the [GroupLens research licence](https://files.grouplens.org/datasets/movielens/ml-32m-README.html). F. Maxwell Harper and Joseph A. Konstan. 2015. [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872). All published features aggregate interactions by series.
- [MovieTweetings](https://github.com/sidooms/MovieTweetings), MIT. Coverage audit only; no account mappings collected.
- [IMDb non-commercial datasets](https://data.imdb.com/non-commercial-datasets/) and [terms for software use](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX). Only aggregate evaluation metrics are reported in public.
'''
    (ROOT/'AUDIENCE_STUDY.md').write_text(text,encoding='utf-8')
    print(f'Public F1 {primary_score:.3f}, local F1 {local_score:.3f}. Wrote audience report and figure.')


if __name__ == '__main__':
    main()
