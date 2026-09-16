"""Check the Measurement Science and Technology submission against IOP's stated rules.

Sources, retrieved 2026-09-15:
  iopjournal-guidelines.pdf shipped with the official template
  https://publishingsupport.iopscience.iop.org/journals/measurement-science-and-technology/
  https://publishingsupport.iopscience.iop.org/questions/latex-template/

Encoded rules:
  - use of the iopjournal class, with articletype, title, author, affil, email,
    keywords and abstract in the order the template gives
  - abstract not more than 300 words, complete in itself: no undefined
    abbreviations and no reference to any table, figure, reference or equation
  - ORCID iDs given as 14 digits with hyphen separators via the orcid command
  - back matter present: acknowledgments, funding, author contributions in
    CRediT terms, data availability
  - all files in one flat directory; file names use only a-z, A-Z, 0-9 and
    underscore; every file referenced in the tex must exist with exactly
    matching case
  - MST additionally requires a novelty statement of under 100 words, supplied
    in the submission system rather than in the manuscript
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / 'paper/mst'
TEX = PAPER / 'mst_manuscript.tex'
NOVELTY = PAPER / 'novelty_statement.txt'
ok, bad, warn = [], [], []
src = TEX.read_text(encoding='utf-8')
body = re.sub(r'(?m)^%.*$', '', src)


def chk(cond, msg, soft=False):
    (ok if cond else (warn if soft else bad)).append(msg)


chk(r'\documentclass{iopjournal}' in body, 'uses the official iopjournal class')
for cmd in ('articletype', 'title', 'author', 'affil', 'email', 'keywords'):
    chk(re.search(r'\\' + cmd + r'[\{\[]', body) is not None, f'{cmd} command present')

ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', body, re.S)
chk(ab is not None, 'abstract environment present')
if ab:
    a = ab.group(1).strip()
    n = len(a.split())
    chk(n <= 300, f'abstract is {n} words (IOP: normally not more than 300)')
    chk(not re.search(r'\\cite|\\ref|\\eqref|figure~|table~|equation~', a, re.I),
        'abstract refers to no table, figure, reference or equation')
    abbr = sorted({w for w in re.findall(r'\b[A-Z]{2,}\b', a)})
    chk(not abbr, 'abstract contains no undefined abbreviations' + ('' if not abbr else f' (found {", ".join(abbr)})'))

orcids = re.findall(r'\\orcid\{([^}]*)\}', body)
chk(len(orcids) > 0, f'{len(orcids)} ORCID iDs supplied via the orcid command')
badid = [o for o in orcids if not re.fullmatch(r'\d{4}-\d{4}-\d{4}-\d{3}[\dX]', o)]
chk(not badid, 'every ORCID iD is 14 digits with hyphen separators'
    + ('' if not badid else f'; malformed or placeholder: {", ".join(badid)}'), soft=True)

authors = re.search(r'\\author\{(.*?)\n\n', body, re.S)
n_auth = len(re.findall(r'\\orcid', authors.group(1))) if authors else 0
n_affil = len(re.findall(r'\\affil\{', body))
chk(n_auth >= 1, f'{n_auth} authors in the author block')
chk(n_affil >= 2, f'{n_affil} affil blocks (one per institution plus the corresponding-author note)')
chk(re.search(r'\$\^\{?[0-9,\*]*\*[0-9,\*]*\}?\$', body) is not None,
    'corresponding author marked with an asterisk in the author list')
chk(re.search(r'Author to whom any correspondence should be addressed', body) is not None,
    'corresponding-author affil line present, as the template requires')

for cmd, label in (('ack', 'acknowledgments'), ('funding', 'funding'),
                   ('roles', 'author contributions (CRediT)'), ('data', 'data availability')):
    chk(re.search(r'\\' + cmd + r'\{', body) is not None, f'back matter present: {label}')
chk(re.search(r'artificial intelligence', body, re.I) is not None,
    'use of AI disclosed in the acknowledgments')
chk(re.search(r'conflict of interest|competing interest', body, re.I) is not None,
    'conflict-of-interest statement present')

labels = set(re.findall(r'\\label\{([^}]+)\}', body))
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', body))
chk(refs <= labels, 'all cross-references resolve'
    + ('' if refs <= labels else f'; dangling: {", ".join(sorted(refs - labels))}'))

nbib = len(re.findall(r'\\bibitem', body))
cited = {int(x) for g in re.findall(r'\[([0-9,\s\-]+)\]', body) for part in g.split(',')
         for x in ([part.strip()] if '-' not in part else [])
         if x.strip().isdigit()}
ranges = [tuple(int(v) for v in m) for m in re.findall(r'\[(\d+)\s*--\s*(\d+)\]', body)]
for lo, hi in ranges:
    cited |= set(range(lo, hi + 1))
missing = sorted(i for i in range(1, nbib + 1) if i not in cited)
chk(nbib > 0, f'{nbib} bibliography entries')
chk(not missing, 'every bibliography entry is cited in the text'
    + ('' if not missing else f'; uncited: {missing}'), soft=True)

figs = set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', body))
for f in sorted(figs):
    p = PAPER / f
    chk(p.exists(), f'figure file present with exactly matching name: {f}')
for f in ('iopjournal.cls', 'orcid.pdf'):
    chk((PAPER / f).exists(), f'template file present: {f}')

badnames = [p.name for p in PAPER.iterdir() if p.is_file() and not re.fullmatch(r'[A-Za-z0-9_]+\.[A-Za-z0-9]+', p.name)]
chk(not badnames, 'all file names use only a-z, A-Z, 0-9 and underscore'
    + ('' if not badnames else f'; offending: {", ".join(badnames)}'))
subdirs = [p.name for p in PAPER.iterdir() if p.is_dir()]
chk(not subdirs, 'submission directory is flat, no subfolders'
    + ('' if not subdirs else f'; found: {", ".join(subdirs)}'))

from collections import Counter
envs, ends = re.findall(r'\\begin\{([a-zA-Z*]+)\}', body), re.findall(r'\\end\{([a-zA-Z*]+)\}', body)
unb = dict((Counter(envs) - Counter(ends)) | (Counter(ends) - Counter(envs)))
chk(not unb, 'all LaTeX environments balanced' + ('' if not unb else f'; {unb}'))
chk(body.count('{') == body.count('}'), f'braces balanced ({body.count("{")} open, {body.count("}")} close)')

if NOVELTY.exists():
    nv = [ln for ln in NOVELTY.read_text(encoding='utf-8').splitlines() if ln.strip() and not ln.startswith('#')]
    w = len(' '.join(nv).split())
    chk(w < 100, f'novelty statement is {w} words (MST requires under 100)')
else:
    chk(False, 'novelty_statement.txt present (MST requires a statement under 100 words)')

ph = re.findall(r'<<[^>]{0,90}', body)
chk(not ph, f'{len(ph)} placeholders still to fill before submission', soft=True)

words = len(re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', body).split())
chk(words > 0, f'approximately {words} source words of manuscript text')

print(f'PASS {len(ok)}   FAIL {len(bad)}   WARN {len(warn)}\n')
for m in ok:
    print('  [ok]   ', m)
for m in warn:
    print('  [warn] ', m)
for m in bad:
    print('  [FAIL] ', m)
if ph:
    print('\nplaceholders to fill:')
    for p in ph:
        print('    ', ' '.join(p.split())[:80])
sys.exit(1 if bad else 0)
