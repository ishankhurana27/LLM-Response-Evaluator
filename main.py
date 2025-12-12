import argparse
from src.evaluator import Evaluator
import json

parser = argparse.ArgumentParser()
parser.add_argument('--conv', required=True)
parser.add_argument('--ctx', required=True)
parser.add_argument('--out', default='report.json')
args = parser.parse_args()

ev = Evaluator()
ev.load_data(args.conv, args.ctx)
ev.build_index()
report = ev.evaluate()
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print('Saved report to', args.out)