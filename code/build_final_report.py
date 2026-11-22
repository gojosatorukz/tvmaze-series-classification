"""Build the English research report from recorded results and reviewed case notes."""
from pathlib import Path
import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/final'
WIDTH = A4[0] - 100


def main():
    summary = json.loads((OUT / 'analysis_summary.json').read_text())
    narrative = json.loads((OUT / 'reviewed_findings.json').read_text())
    coverage = json.loads((OUT / 'text_coverage.json').read_text())
    tokens = json.loads((OUT / 'tokenization_audit.json').read_text())
    models = {row['family']: row for row in summary['models']}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', fontName='Times-Bold', fontSize=25,
                              leading=29, spaceAfter=12, textColor=colors.HexColor('#152936')))
    styles.add(ParagraphStyle(name='ReportBody', fontName='Times-Roman', fontSize=11,
                              leading=15, spaceAfter=8, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle(name='ReportHead', fontName='Helvetica-Bold', fontSize=14,
                              leading=18, spaceBefore=15, spaceAfter=9, keepWithNext=True))
    styles.add(ParagraphStyle(name='ReportSmall', fontName='Helvetica', fontSize=8.4,
                              leading=11.5, spaceAfter=7))
    styles.add(ParagraphStyle(name='ReportCell', fontName='Helvetica', fontSize=8.4, leading=11))
    styles.add(ParagraphStyle(name='ReportCode', fontName='Courier', fontSize=8.1, leading=11.5,
                              spaceAfter=8))
    story = []

    def paragraph(value, style='ReportBody'):
        story.append(Paragraph(value, styles[style]))

    def heading(value):
        paragraph(value, 'ReportHead')

    def table(rows, widths):
        wrapped = [[Paragraph(escape(str(value)), styles['ReportCell']) for value in row] for row in rows]
        object = Table(wrapped, colWidths=widths, repeatRows=1, hAlign='LEFT')
        object.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E7EEF0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), .45, colors.HexColor('#D9D9D9')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        story.extend([object, Spacer(1, 10)])

    def figure(filename, caption, width=WIDTH):
        path = OUT / 'figures' / filename
        with PILImage.open(path) as bitmap:
            height = width*bitmap.height/bitmap.width
        story.append(Image(str(path), width=width, height=height))
        paragraph(caption, 'ReportSmall')

    paragraph('Does a plot synopsis improve TV-series rating classification?', 'ReportTitle')
    paragraph('Machine Learning Algorithms - Final project - SE-2418<br/>'
              'Bekzat Rashiduly, Gapar Galym, Aibek Galmbekov<br/>22 November 2026', 'ReportSmall')
    paragraph(narrative['abstract'])
    heading('1. Question and contribution')
    paragraph('The study asks whether a series synopsis adds predictive information about a TVmaze '
              'rating of at least 8/10 when structured metadata are already available. The target is a binary '
              'label from an observed aggregate rating. The practical prototype returns a class and model '
              'score for a new set of inputs. This is a catalogue-level classification study.')
    paragraph('The Midterm established a collected dataset and structured-feature benchmarks. The Endterm '
              'added ensembles, an unsupervised analysis and an MLP. The Final adds a frozen pretrained '
              'text encoder, simple learned text controls, exhaustive model searches, case-level diagnostics '
              'and a working saved-model app. These additions test the proposed explanation that metadata '
              'omit story information.')
    heading('2. Data, rights and matching')
    paragraph('The original collector requested 60 pages from the public TVmaze API on 6 October 2026. '
              'The raw snapshot contains 14,055 series. Cleaning retained 2,339 rated English scripted '
              'series, including 404 with ratings at least 8. The dataset includes 1,732 English scripted '
              'series with unknown ratings that were excluded because their labels were unavailable. '
              'One row represents one TV series. IDs are unique. [1]')
    table([['Partition', 'Series', 'Positive labels', 'Missing synopses'],
           ['Training', '1,871', '323', '0'], ['Test benchmark', '468', '81', '0']],
          [WIDTH*.37, WIDTH*.17, WIDTH*.23, WIDTH*.23])
    paragraph('TVmaze licenses its API data under CC BY-SA 4.0. The submission credits TVmaze, preserves '
              'the raw responses and records SHA-256 hashes. The cleaned text removes HTML markup. '
              'Each synopsis is already linked to the same show_id as its metadata and label. '
              'No additional object matching was needed. All 2,339 objects have a synopsis. [1]')
    paragraph('The seven registered inputs are premiere year, episode runtime, network, web channel, '
              'broadcaster/platform country, distribution and genres. Rating creates the label and is '
              'excluded from model inputs. IDs and titles support auditing. The Final additionally derives '
              'genre count, genre combinations, premiere decade and runtime buckets. The categories describe '
              'the catalogue snapshot. They do not establish what was known before a series premiered.')
    paragraph('The training descriptions average 75.6 words. MiniLM uses the first 256 WordPiece tokens. '
              f'This truncates {tokens["splits"][0]["over_256_tokens"]} of 1,871 training texts and '
              f'{tokens["splits"][1]["over_256_tokens"]} of 468 test texts. An explicit scan found '
              'no IMDb/TVmaze rating mentions or numeric 8/10-style rating phrases in the descriptions. '
              'The raw and cleaned data, saved IDs and split hashes reproduce the earlier stages.')
    heading('3. Existing approaches and design choices')
    paragraph('Sparse text classification represents words or word pairs with TF-IDF and fits a linear '
              'classifier. This provides a compact reference trained on the project data. The vocabulary '
              'and IDF are fitted within each training partition. [2]')
    paragraph('Sentence-BERT studies sentence embeddings that support efficient semantic comparison. '
              'MiniLM studies transformer compression by self-attention distillation. The selected '
              'all-MiniLM-L6-v2 encoder produces 384-dimensional representations. This study uses its '
              'frozen, six-layer, quantised ONNX model with attention-masked mean pooling and L2 '
              'normalisation. A logistic classifier learns the binary rating task from those vectors. [3-5]')
    paragraph('The model revision and downloaded-file hashes are recorded in assets/minilm/manifest.json. '
              'The pretrained weights use Apache-2.0. They run locally through ONNX Runtime. A source '
              'model card and license accompany the files. The API data license and model license '
              'apply to their respective artefacts.')
    heading('4. Evaluation and full searches')
    paragraph('The saved stratified 80/20 split contains 1,871 training objects and 468 test objects. '
              'Five saved outer folds partition the training set. An independent three-fold inner '
              'GridSearchCV chooses settings inside each outer fitting partition. The outer fold '
              'then evaluates the selected pipeline. A separate five-fold search on all training '
              'objects selects deployment settings. Numeric imputation, scaling, category encoding, '
              'TF-IDF and latent projections fit inside the corresponding training partition. [6]')
    paragraph('Positive-class F1 is the primary comparison metric. Highly rated series form 17.3% '
              'of the cohort. A majority-only predictor achieves about 83% accuracy with zero '
              'positive recall. Precision describes the share of a shortlist that is truly highly rated. '
              'Recall describes the share of highly rated series recovered. ROC-AUC and average '
              'precision describe ranking. Reported cross-validation spreads are sample standard '
              'deviations across five outer folds.')
    table([['Search', 'Varied dimensions', 'Cartesian combinations'],
           ['Random Forest', 'Trees, depth, leaf size, feature fraction', '4 x 4 x 4 x 4 = 256'],
           ['Gradient Boosting', 'Trees, learning rate, depth, leaf size', '4 x 4 x 4 x 4 = 256'],
           ['MLP', 'Architecture, L2 penalty, learning rate', '4 x 4 x 4 = 64']],
          [WIDTH*.25, WIDTH*.47, WIDTH*.28])
    paragraph('Random Forest searches 100, 200, 400 and 600 trees. Gradient Boosting searches 50, '
              '100, 200 and 300 stages. The full parameter lists and the positions of selected '
              'settings are in Appendix A and the exported search tables. The neural grid compares '
              'hidden layers (32), (64), (64, 32) and (128, 64). It varies L2 alpha and learning rate '
              'over four values each. Every combination is evaluated.')
    paragraph('The text study compares metadata-only, text-only and combined inputs with logistic '
              'regression. Its regularisation grid is C in {0.001, 0.01, 0.1, 1}. Combined models '
              'also tune the text-block weight in {0.3, 1, 3, 6}. TF-IDF uses word unigrams and '
              'bigrams, min_df=2, max_df=0.95 and at most 20,000 features. A 48-component '
              'TruncatedSVD control tests a compact latent representation. MiniLM weights remain '
              'fixed while the task-specific classifier is fitted.')
    paragraph('An additional metadata experiment tests cross-fitted target encoding for sparse '
              'categories, with smoothing in {5, 10, 20, 50}. Encoding training rows uses an internal '
              'five-fold procedure so each row is represented using other encoder folds. Validation '
              'and new inputs use statistics from the fitting data. This check tests an alternative '
              'to collapsing rare categories into one group. It is also paired with MiniLM. [7]')
    paragraph('The test IDs were already reported at the Midterm and Endterm. They are a reused '
              'course benchmark. The new model choices are recorded from training cross-validation '
              'before Final test predictions are produced. The same objects, labels and metrics '
              'support the comparison. Paired stratified bootstrap intervals resample the fixed '
              'test objects 2,000 times with seed 42. They condition on the fitted models and '
              'do not account for retraining or dependence between related franchises.')
    heading('5. Results and incremental value of text')
    figure('representation_comparison.png', 'Figure 1. Outer training CV mean and sample SD, alongside the reused test benchmark. All models use the saved objects and split.')
    rows = [['Input representation', 'Outer CV F1 +/- SD', 'Test F1', 'Precision', 'Recall']]
    for key in ['original_metadata', 'engineered_metadata', 'tfidf_text', 'tfidf_combined',
                'lsa_text', 'lsa_combined', 'minilm_text', 'minilm_combined',
                'target_encoded_metadata', 'target_minilm_combined', 'random_forest',
                'gradient_boosting', 'mlp', 'svm_reference', 'endterm_rf_reference',
                'dummy_majority', 'dummy_stratified']:
        if any(row[0] == models[key]['name'] for row in rows[1:]):
            continue
        row = models[key]
        rows.append([row['name'], f'{row["cv_f1_mean"]:.3f} +/- {row["cv_f1_sd"]:.3f}',
                     f'{row["test_f1"]:.3f}',
                     str(Decimal(str(row['test_precision'])).quantize(Decimal('.001'), rounding=ROUND_HALF_UP)),
                     f'{row["test_recall"]:.3f}'])
    table(rows, [WIDTH*.38, WIDTH*.25, WIDTH*.12, WIDTH*.125, WIDTH*.125])
    for value in narrative['results']:
        paragraph(value)
    heading('6. Neural architecture and learning curves')
    figure('architecture_comparison.png', 'Figure 2. Best full-training selection-CV configuration within each architecture. These are tuning scores. The outer MLP estimate evaluates the complete architecture search.')
    figure('mlp_learning.png', 'Figure 3. Fitting and internal-validation curves for the chosen MLP. The dashed line marks the restored absolute validation-loss minimum.')
    mlp = summary['mlp']
    paragraph(f'The chosen hidden layers are {mlp["best_params"]["hidden_layer_sizes"]}, '
              f'with alpha {mlp["best_params"]["alpha"]} and learning rate '
              f'{mlp["best_params"]["learning_rate_init"]}. The network has {mlp["parameters"]:,} '
              f'parameters, including biases, and {mlp["encoded_inputs"]} encoded inputs. '
              'It uses ReLU, Adam, batches of 64 and class-balanced fitting weights. '
              f'The internal split has {mlp["fitting_rows"]:,} fitting objects and '
              f'{mlp["validation_rows"]} validation objects.')
    paragraph(f'The validation-loss minimum occurs at epoch {mlp["best_epoch"]}, with loss '
              f'{mlp["best_validation_loss"]:.4f}. At that epoch, fitting loss is '
              f'{mlp["best_train_loss"]:.4f}, giving a gap of {mlp["gap_at_best"]:.4f}. '
              f'By epoch {mlp["stopped_epoch"]}, fitting loss reaches '
              f'{mlp["last_train_loss"]:.4f} while validation loss reaches '
              f'{mlp["last_validation_loss"]:.4f}. The gap widens to {mlp["gap_at_stop"]:.4f}. '
              'This divergence indicates increasing overfitting. Later fitting improvement failed '
              'to improve validation probabilities. Patience allows the procedure to observe the '
              'trend. The saved model restores the weights at the absolute validation minimum.')
    paragraph('Each outer MLP fit repeats its own internal stratified validation split and checkpoint '
              'selection. The final checkpoint retains preprocessing fitted on its fitting subset. '
              'Validation log loss selects the epoch. Positive-class F1 evaluates the resulting '
              'thresholded predictions. The earlier minimum-improvement rule was corrected so a '
              'small new absolute minimum is always retained.')
    heading('7. Errors and model sensitivity')
    figure('confusion_matrices.png', 'Figure 4. True classes are rows and predicted classes are columns. The three deployed input variants are compared on the same 468 test series.')
    for case in narrative['cases']:
        paragraph('<b>'+escape(case['name'])+'</b>')
        paragraph(case['text'])
    paragraph(narrative['additional_error_note'])
    paragraph('The evidence file records feature replacements using training reference values and '
              'sentence-occlusion scores. These interventions quantify model sensitivity. The '
              'dataset lacks direct measurements of acting, editing, writing quality and audience '
              'preferences. The observed aggregate rating also lacks vote counts. These omissions '
              'limit the explanations of the audience ratings themselves.')
    heading('8. Local app and reproducibility')
    paragraph('The Flask app loads saved model pipelines, accepts metadata and an English synopsis, '
              'and returns predictions from metadata, text and combined models. The primary model '
              'is chosen using training CV. Blank numeric inputs use saved training medians. '
              'Unknown categories follow the fitted encoding policy. When the synopsis is empty, '
              'the app returns the metadata prediction. Invalid numeric values receive a clear '
              'input error. Labels and IDs supplied as extra input fields are ignored by feature '
              'construction. Model scores are uncalibrated.')
    paragraph('From a complete extracted submission, create a Python 3.12 environment, install '
              'requirements-final.txt and run python app.py. The app binds to 127.0.0.1:8765. '
              'The included models, data and MiniLM weights support local inference. The full '
              'reproduction route is python code/run_final.py --fresh --workers 4. It creates a '
              'separate directory under .runtime, preserving the supplied results, and runs the exhaustive '
              'searches and exports the comparison and diagnostics. Completed outer-fold results '
              'are checkpointed. The verification route is python code/run_final.py --verify-only. '
              'It checks hashes, grid coverage, loaded-model predictions and the neural checkpoint.')
    paragraph('Seeds and numerical-library threads are fixed. The exact tested versions and model '
              'revision are recorded. Full searches take substantially longer than saved-model '
              'inference. Differences in hardware or numerical libraries can produce small '
              'floating-point variation. The test-prediction verification and saved search results '
              'make such differences inspectable.')
    heading('9. Limitations and conclusion')
    for value in narrative['limitations']:
        paragraph(value)
    paragraph(narrative['conclusion'])
    heading('10. Contributions and tools')
    paragraph('Bekzat Rashiduly is the technical project owner responsible for data collection and '
              'continuity, implementation, experimental design, model evaluation, the local app '
              'and report. Gapar Galym and Aibek Galmbekov are scheduled to participate in the '
              'course presentation. The project is submitted as a team course project, with the '
              'technical development led individually by Bekzat.')
    paragraph('OpenAI Codex assisted with coding, experiment execution, analysis, figures, report '
              'and presentation preparation, and verification. Python, pandas, NumPy, scikit-learn, '
              'Matplotlib, ONNX Runtime, Hugging Face Tokenizers, Flask and Jupyter support the '
              'workflow. The course permits AI tools and requires their uses to be declared. '
              'Each presenter remains responsible for understanding and explaining the work.')
    heading('References')
    references = [
        '[1] TVmaze. Public API and licensing. https://www.tvmaze.com/api',
        '[2] scikit-learn. Text feature extraction and sparse document classification. https://scikit-learn.org/stable/modules/feature_extraction.html',
        '[3] Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. https://arxiv.org/abs/1908.10084',
        '[4] Wang, W. et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. https://arxiv.org/abs/2002.10957',
        '[5] Sentence Transformers. all-MiniLM-L6-v2 model card and pinned model files. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
        '[6] scikit-learn. Nested versus non-nested cross-validation. https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html',
        '[7] scikit-learn. Target Encoder internal cross fitting. https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html',
    ]
    for value in references:
        paragraph(escape(value), 'ReportSmall')
    story.append(PageBreak())
    heading('Appendix A. Search settings and boundary positions')
    rows = [['Model / parameter', 'Tested values', 'Selected', 'Position']]
    for item in summary['parameter_boundaries']:
        parameter = item['parameter'].replace('model__', '')
        family = item['family'].replace('_', ' ')
        rows.append([family+' / '+parameter, item['tested'], item['selected'], item['position']])
    table(rows, [WIDTH*.31, WIDTH*.34, WIDTH*.13, WIDTH*.22])
    paragraph('The exported full_train_grid.csv files contain every combination and its fold scores. '
              'The outer search tables record the complete independent inner search for each saved '
              'outer fold. Boundary positions describe the declared grid. An optimum at a boundary '
              'supports further range investigation in a later experiment.')
    heading('Appendix B. Response to earlier feedback')
    for item in narrative['feedback']:
        paragraph('<b>'+escape(item['issue'])+'</b>. '+item['response'])
    paragraph('The full Endterm feedback is attached as docs/TEACHER_FEEDBACK.txt. It identifies '
              'the earlier Midterm concerns about general error examples and templated conclusions. '
              'The current Midterm assignment view does not expose the complete original comment '
              'text. The responses above address the issues identified in the Endterm feedback.', 'ReportSmall')
    folder = ROOT / 'report'
    folder.mkdir(exist_ok=True)
    document = SimpleDocTemplate(str(folder / 'Final_Report.pdf'), pagesize=A4,
                                  leftMargin=50, rightMargin=50, topMargin=48, bottomMargin=45,
                                  title='TVmaze: incremental value of plot synopses', author='Bekzat Rashiduly')
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#677B84'))
        canvas.drawString(50, 25, 'TVmaze - Final research report - November 2026')
        canvas.drawRightString(A4[0]-50, 25, str(doc.page))
        canvas.restoreState()
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(folder / 'Final_Report.pdf')


if __name__ == '__main__':
    main()
