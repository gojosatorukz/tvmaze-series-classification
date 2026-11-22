# Does training on more series get F1 to 0.80?

The highest observed F1 on the earlier 468-title benchmark remains **0.663** with IMDb title rating and vote count. The same frozen recipe reached **0.270** on 379 different series. Training on another 1,118 TVmaze titles and trying six estimators across three input blocks did not reach 0.80 on either evaluation.

## Data and protocol

- Earlier benchmark. 468 titles sampled from the original October TVmaze cohort, with 81 positive labels. The original fitting set has 1,871 titles and 323 positives.
- Additional fitting data. TVmaze index pages 100–199 yield 1,118 eligible titles, with 30 positives. The expanded fitting set contains 2,989 titles. Premiere year median is 2019 in the new fitting slice.
- New catalogue evaluation. TVmaze index pages 200–239 yield 379 eligible titles and 10 positives. IMDb title ratings cover 361 of those titles. The median premiere year is 2022. The first catalogue has a median of 2010.

The page ranges were set before testing. The collector excludes overlaps by TVmaze ID, IMDb ID and normalized title with premiere year, including overlaps with the previous catalogue slice. All data follow the same English scripted and premiere-date eligibility rules. The positive class is TVmaze rating ≥ 8 in each snapshot. The November slices have a smaller share of positives than the October catalogue.

The grids compare logistic regression, spline logistic regression, Extra Trees, two histogram boosting classifiers and a histogram rating regressor. Inputs range from the IMDb series rating and vote count, through IMDb episode-rating summaries, to additional TVmaze metadata. IMDb joins use exact IDs. The models receive ratings for the same series from a separate website and its episodes. The evaluation is retrospective.

A common pool of out-of-fold scores from both training catalogues selects each candidate's threshold. Estimators at the original fitting size learn their coefficients or trees from the 1,871 original titles only; extra labels are available for threshold calibration and candidate comparison. The expanded fits also learn from the extra titles. The candidate criterion is the mean of the two training-catalogue F1 values. All candidate and threshold choices are recorded in `results/audience_scaleup/frozen_selection.json`. The two logistic rows use the same features and estimator with each threshold calibrated by the same rule. The matched-recipe comparison was added as a secondary analysis after the first scale-up evaluation.

## Results

| Pipeline | Fitting rows | Earlier benchmark F1 | New catalogue F1 | New precision | New recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Final gradient boosting | 1,871 | 0.350 | 0.092 | 0.055 | 0.300 |
| Earlier IMDb rating and vote count model | 1,871 | 0.663 | 0.270 | 0.185 | 0.500 |
| Logistic regression, selected from the original fitting size | 1,871 | 0.641 | 0.174 | 0.154 | 0.200 |
| Same logistic regression recipe with extra training titles | 2,989 | 0.621 | 0.200 | 0.200 | 0.200 |
| Boosted rating regression, selected from the expanded fitting size | 2,989 | 0.621 | 0.250 | 0.333 | 0.200 |

The earlier IMDb model retrieves 5 of the 10 positives on the new catalogue, adds 22 false positives and misses 5. The new slice has 2.6% positives, compared with 17.3% on the old benchmark. With so few true positives, even a small number of false positives materially changes F1. Increasing the fitting size while fixing the logistic model recipe changes fresh F1 from **0.174 to 0.200**. The tests estimate performance on these catalogue slices, with substantial uncertainty on the ten-positive sample.

For a separate diagnostic, every possible score threshold was evaluated using the known test labels of each IMDb-assisted model. The highest F1 even with this advantage was **0.667** on the old benchmark and **0.370** on the new catalogue. This is a ceiling for these fitted rankings with a single score threshold on these samples. The diagnostic does not give an honest threshold to deploy or a ceiling for new models and data.

## Reproduce

The included `data/audience_scaleup/` snapshots store TVmaze catalogue pages and hashes. The results ledger, full 36-candidate table and evaluation include aggregate metrics only. For the IMDb-assisted study, obtain the official datasets for personal non-commercial use and place the files at `imdb-title-ratings.tsv.gz` and `imdb-title-episode.tsv.gz` in a local directory. Their snapshot hashes are in `results/audience_scaleup/frozen_selection.json`. New copies of the IMDb files can produce different numbers.

With the Final environment active and all three project archives extracted if starting from GitHub, run:

```bash
python code/audience_scaleup.py /path/to/local/imdb-data
```

The command refits the models and writes local fitted binaries in `results/audience_scaleup/`. IMDb data and those fitted binaries remain local. Reproducing the high-level slate requires the supplied frozen audience selection and the original saved course model.

Sources: [TVmaze API and CC BY-SA terms](https://www.tvmaze.com/api), [IMDb non-commercial datasets](https://data.imdb.com/non-commercial-datasets/), [IMDb use conditions](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX). Information courtesy of IMDb ([IMDb](https://www.imdb.com)).
