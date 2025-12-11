import csv
from pathlib import Path
import statistics

CSV = Path('data/batch_results.csv')
OUT = Path('docs/benchmark_results.md')

if not CSV.exists():
    print('No batch_results.csv found. Run batch eval first (scripts/run_batch_eval.ps1).')
    exit(1)

rows = []
with open(CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# compute stats (convert strings to floats where applicable)
relevance = [float(r['relevance']) for r in rows]
completeness = [float(r['completeness']) for r in rows]
factuality = [float(r.get('factuality', r.get('factuality_avg', 0))) for r in rows]

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('# Benchmark Results\\n\\n')
    f.write(f'*Total evaluated files:* {len(rows)}\\n\\n')
    if rows:
        f.write('## Summary Statistics\\n\\n')
        f.write(f'- Average relevance: {statistics.mean(relevance):.3f}\\n')
        f.write(f'- Average completeness: {statistics.mean(completeness):.3f}\\n')
        f.write(f'- Average factuality: {statistics.mean(factuality):.3f}\\n\\n')

        f.write('## Detailed results\\n\\n')
        f.write('| chat_file | relevance | completeness | factuality | verdict | total_tokens |\\n')
        f.write('|---|---:|---:|---:|---|---:|\\n')
        for r in rows:
            f.write(f"| {r['chat_file']} | {r['relevance']} | {r['completeness']} | {r.get('factuality', r.get('factuality_avg', ''))} | {r['verdict']} | {r.get('total_tokens','')} |\\n")
    else:
        f.write('No rows found in CSV.\\n')

print('Wrote', OUT)
