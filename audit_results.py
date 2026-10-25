"""Read previous verified local results and produce an evidence audit. Never invent measurements."""
import json
import argparse
from pathlib import Path

EXPECTED = {
    'lambdamart_metrics.json': ['BM25_NDCG@10', 'MiniLM_NDCG@10', 'LambdaMART_NDCG@10'],
    'reliability_metrics.json': ['BM25_NDCG@10', 'MiniLM_NDCG@10', 'LambdaMART_NDCG@10', 'Adaptive_NDCG@10'],
    'end_to_end_metrics.json': ['methods', 'catalog_products', 'total_evaluated_queries'],
    'phase8_summary.json': ['sampled_queries'],
}

def inspect_results(directory):
    directory = Path(directory)
    report = {'source_directory': str(directory), 'checks': {}, 'warnings': [
        'The original 30-query test split has been consulted repeatedly; reserve a fresh final test split for new claims.',
        'Judgment-based NDCG treats unjudged products as zero and is biased by incomplete relevance labels.',
        'Previously observed adaptive ranking underperformed the strongest baseline; do not claim it improved performance.',
    ]}
    for file, keys in EXPECTED.items():
        path = directory / file
        if not path.exists():
            report['checks'][file] = {'status': 'missing'}
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            missing = [k for k in keys if k not in data]
            report['checks'][file] = {'status': 'ok' if not missing else 'missing_keys', 'missing_keys': missing, 'data': data}
        except (OSError, ValueError) as e:
            report['checks'][file] = {'status': 'invalid', 'error': str(e)}
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results')
    parser.add_argument('--output', default='results/evidence_audit.json')
    args = parser.parse_args()
    audit = inspect_results(args.results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k: v.get('status') for k,v in audit['checks'].items()}, indent=2))
    print('Saved:', output)