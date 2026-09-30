"""Markdown table of N-SFLM vs N-MDLM results produced by run_compare."""
import argparse
import json
import sys
from pathlib import Path

from sfmlm.config import Config
from sfmlm.constants.experiment import MODEL_LABELS
from sfmlm.constants.neuromorphic import MODES, MODE_FLOAT

HEADER = ('| model | S | mode | K | acc | acc - float | spikes/tok (bin) '
          '| spikes/tok (NAF) | rate/tok | sigma | row-active | clip '
          '| ICMS thr x dense | ICMS E/tok x dense | OCMS thr x dense |')


def _rows(results):
  float_acc = {(r['model'], r['steps']): r['runs'][0]['accuracy']
               for r in results if r['mode'] == MODE_FLOAT}
  order = {m: i for i, m in enumerate(MODES)}
  rows = []
  for r in sorted(results, key=lambda r: (r['model'], r['steps'],
                                          order[r['mode']])):
    base = float_acc.get((r['model'], r['steps']))
    for run in r['runs']:
      rf = run['roofline']
      thr = {hw: (rf[hw].get('spiking', rf[hw]['dense'])['throughput_tok_s']
                  / rf[hw]['dense']['throughput_tok_s'])
             for hw in ('ICMS', 'OCMS')}
      energy = (rf['ICMS'].get('spiking', rf['ICMS']['dense'])
                ['energy_per_token_j']
                / rf['ICMS']['dense']['energy_per_token_j'])
      s = run.get('spikes')
      delta = '' if base is None else f'{run["accuracy"] - base:+.4f}'
      cells = [MODEL_LABELS.get(r['model'], r['model']), str(r['steps']),
               r['mode'] + ('+head' if r.get('include_head') else ''),
               '' if run['K'] is None else f'{run["K"]:g}',
               f'{run["accuracy"]:.4f}', delta]
      if s:
        cells += [f'{s["spikes_binary_per_token"]:.3g}',
                  f'{s["spikes_naf_per_token"]:.3g}',
                  f'{s["rate_count_per_token"]:.3g}',
                  f'{s["sigma"]:.4f}', f'{s["row_active"]:.3f}',
                  f'{s["clip_rate"]:.1e}']
      else:
        cells += [''] * 6
      cells += [f'{thr["ICMS"]:.2f}', f'{energy:.3f}', f'{thr["OCMS"]:.2f}']
      rows.append('| ' + ' | '.join(cells) + ' |')
  return rows


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--results-dir', type=Path, default=Config.RESULTS_DIR)
  p.add_argument('--pattern', default='*.json')
  p.add_argument('--out', type=Path, default=None)
  args = p.parse_args(argv)
  results = []
  for f in sorted(args.results_dir.glob(args.pattern)):
    try:
      results.append(json.loads(f.read_text()))
    except (json.JSONDecodeError, OSError) as e:
      print(f'[skip] {f.name}: {e}', file=sys.stderr)
  if not results:
    raise SystemExit(f'no results matching {args.pattern} in '
                     f'{args.results_dir}')
  sep = '|' + '---|' * HEADER.count(' | ') + '---|'
  table = '\n'.join([HEADER, sep, *_rows(results)])
  print(table)
  if args.out:
    args.out.write_text(table + '\n')


if __name__ == '__main__':
  main()
