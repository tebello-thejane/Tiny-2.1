#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.15"
# dependencies = []
# ///
"""More table facts from public datasets.
  facts/geo3.tsv      subregion, borders, domain   (mledoze/countries)
  facts/elements.tsv  symbol, number, phase, kind  (Bowserinator/Periodic-Table-JSON)
Usage: ./extra_facts.py"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo2 import ascii_only

HERE = os.path.dirname(os.path.abspath(__file__))

caps = {}
for line in open(os.path.join(HERE, 'facts/geo.tsv')):
    r, s, a = line.rstrip('\n').split('\t')
    caps.setdefault(s, a)

data = json.load(open(os.path.join(HERE, 'mledoze/countries.json'), encoding='utf-8'))
name3 = {}
for c in data:
    n = ascii_only(c['name']['common'])
    name3[c['cca3']] = {'Turkiye': 'Turkey'}.get(n, n)
rows = []
for c in data:
    if not c.get('unMember'):
        continue
    n = name3[c['cca3']]
    if n not in caps:
        continue
    if c.get('subregion'):
        rows.append(('subregion', n, c['subregion']))
    b = [name3[x] for x in c.get('borders', []) if x in name3 and name3[x] in caps]
    rows.append(('borders', n, '|'.join(b) if b else '-'))
    if c.get('tld'):
        rows.append(('domain', n, c['tld'][0]))
with open(os.path.join(HERE, 'facts/geo3.tsv'), 'w') as f:
    for r in rows:
        f.write('\t'.join(r) + '\n')
print('geo3', len(rows))

els = json.load(open(os.path.join(HERE, 'ptable/PeriodicTableJSON.json')))['elements']
rows = []
for e in els:
    if e['number'] > 118:
        continue
    n = e['name']
    rows.append(('symbol', n, e['symbol']))
    rows.append(('atomic', n, str(e['number'])))
    cat = e['category']
    if cat.startswith('unknown'):
        continue
    kind = ('a noble gas' if cat == 'noble gas' else 'a metalloid' if cat == 'metalloid'
            else 'a nonmetal' if 'nonmetal' in cat else 'a metal')
    rows.append(('kind', n, kind))
    if e['number'] <= 100 and e.get('phase') in ('Solid', 'Liquid', 'Gas'):
        rows.append(('phase', n, e['phase'].lower()))
with open(os.path.join(HERE, 'facts/elements.tsv'), 'w') as f:
    for r in rows:
        f.write('\t'.join(r) + '\n')
print('elements', len(rows))
