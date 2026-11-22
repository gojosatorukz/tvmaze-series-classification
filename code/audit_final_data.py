"""Reproduce text coverage, object matching and token-truncation checks."""
from pathlib import Path
import re

from tokenizers import Tokenizer
from endterm import check_continuity
from final_experiments import save_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    clean, train, test, _, _, continuity = check_continuity(ROOT)
    out = ROOT / 'results/final'
    out.mkdir(parents=True, exist_ok=True)
    coverage = {}
    for name, frame in [('all', clean), ('train', train), ('test', test)]:
        lengths = frame.plot_summary.str.split().str.len()
        coverage[name] = {'rows': len(frame), 'missing_synopsis': int(frame.plot_summary.str.strip().eq('').sum()),
                          'mean_words': float(lengths.mean()), 'median_words': float(lengths.median()),
                          'max_words': int(lengths.max())}
    instruction_pattern = re.compile(
        r'ignore\s+(?:(?:all|the)\s+)?(?:previous|prior|above)\s+instructions|'
        r'(?:reveal|print|show)\s+(?:the\s+)?system\s+prompt|'
        r'you\s+are\s+(?:chatgpt|codex)|<\|(?:system|assistant)\|>', re.I)
    suspicious = []
    for row in clean.itertuples():
        for match in instruction_pattern.finditer(row.plot_summary):
            suspicious.append({'show_id': int(row.show_id), 'exact_text': match.group(0),
                               'source': 'data/processed/series_clean.csv: plot_summary'})
    save_json(out / 'text_coverage.json', {
        'coverage': coverage, 'unique_object_ids': int(clean.show_id.nunique()),
        'snapshot_date': '2026-10-06', 'data_license': 'CC BY-SA 4.0',
        'source': 'https://www.tvmaze.com/api',
        'matching': 'Synopsis already stored with the same show_id in the immutable snapshot',
        'continuity': continuity, 'suspicious_instruction_matches': suspicious,
    })
    tokenizer = Tokenizer.from_file(str(ROOT / 'assets/minilm/tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    splits = []
    for name, frame in [('train', train), ('test', test)]:
        lengths = [len(item.ids) for item in tokenizer.encode_batch(frame.plot_summary.tolist())]
        long = sum(length > 256 for length in lengths)
        splits.append({'split': name, 'rows': len(frame), 'max_tokens_before_truncation': max(lengths),
                       'over_256_tokens': long, 'truncation_fraction': long/len(frame)})
    mentions = []
    rating_pattern = re.compile(r'\b(?:IMDb|TVmaze)\s+(?:rating|score)|\b(?:[0-9](?:\.\d+)?)\s*/\s*10\b', re.I)
    for row in clean.itertuples():
        for match in rating_pattern.finditer(row.plot_summary):
            mentions.append({'show_id': int(row.show_id), 'exact_text': match.group(0)})
    save_json(out / 'tokenization_audit.json', {'max_length': 256, 'splits': splits,
                                              'explicit_rating_mention_matches': mentions})
    print('Coverage:', coverage)
    print('Token truncation:', splits)
    print('Instruction-pattern matches:', suspicious)
    print('Explicit rating mentions:', mentions)


if __name__ == '__main__':
    main()
