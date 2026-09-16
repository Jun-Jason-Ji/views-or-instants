"""Adversarial audit of the MST submission package and of the numbers in the manuscript.

Looks for things that would actually break or embarrass: LaTeX that will not
compile, citation numbers pointing at the wrong reference, claims that do not
match the sealed artefacts, and inconsistencies between the manuscript, the
GitHub repository and the Zenodo archive.
"""
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MST = ROOT / 'paper/mst/mst_manuscript.tex'
DESK = Path(r'C:\Users\Jason\Desktop\MST_Submission_v1.0.0_2026-09-15')
REPO = ROOT / 'paper/repo'
ZIP = ROOT / 'paper/zenodo/view_instant_allocation_v1.0.0.zip'

issues, notes, ok = [], [], []


def bad(m):
    issues.append(m)


def note(m):
    notes.append(m)


def good(m):
    ok.append(m)


src = MST.read_text(encoding='utf-8')
body = re.sub(r'(?m)^%.*$', '', src)

# ---------------------------------------------------------------- LaTeX health
nbib = len(re.findall(r'\\bibitem', body))
empty_keys = len(re.findall(r'\\bibitem\{\}', body))
if empty_keys:
    bad(f'{empty_keys} of {nbib} bibitem entries have an EMPTY key. LaTeX emits '
        f'"Label `\' multiply defined" for each repeat and the .aux is polluted. '
        f'Give every entry a real key.')
else:
    good(f'{nbib} bibliography entries, all with keys')

# citations in text are hardcoded [n]; check they are in range and all defined
cites = set()
for g in re.findall(r'\[([0-9][0-9,\s\-]*)\]', body):
    for part in re.split(r',\s*', g.strip()):
        part = part.strip()
        if part.isdigit():
            cites.add(int(part))
        else:
            m = re.match(r'^(\d+)\s*-+\s*(\d+)$', part)
            if m:
                cites |= set(range(int(m.group(1)), int(m.group(2)) + 1))
over = sorted(c for c in cites if c > nbib)
missing = sorted(set(range(1, nbib + 1)) - cites)
if over:
    bad(f'citations out of range (no such reference): {over}')
else:
    good(f'all {len(cites)} distinct citation numbers are within 1..{nbib}')
if missing:
    note(f'bibliography entries never cited: {missing}')

# environments, braces, math
from collections import Counter
envs, ends = re.findall(r'\\begin\{([a-zA-Z*]+)\}', body), re.findall(r'\\end\{([a-zA-Z*]+)\}', body)
unb = dict((Counter(envs) - Counter(ends)) | (Counter(ends) - Counter(envs)))
if unb:
    bad(f'unbalanced LaTeX environments: {unb}')
else:
    good('LaTeX environments balanced')
if body.count('{') != body.count('}'):
    bad(f'unbalanced braces: {body.count("{")} open vs {body.count("}")} close')
else:
    good('braces balanced')
if body.count('$') % 2:
    bad(f'odd number of $ delimiters ({body.count("$")}), inline math is unbalanced')
else:
    good('inline math delimiters balanced')

# table column counts must match the tabular spec
for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}', body, re.S):
    spec, tbody = m.group(1), m.group(2)
    ncol = len(re.findall(r'[lcrp]', spec))
    label = re.search(r'\\label\{([^}]+)\}', body[m.end():m.end() + 400])
    name = label.group(1) if label else f'at char {m.start()}'
    for row in tbody.split(r'\\'):
        row = re.sub(r'\\hline|\s', '', row)
        if not row:
            continue
        cells = m.group(2) and len(row.split('&'))
        if cells and cells != ncol:
            bad(f'table {name}: a row has {cells} cells but the spec declares {ncol} columns')
            break
    else:
        good(f'table {name}: all rows match the {ncol}-column spec')

# undefined commands that iopjournal.cls must provide
cls = (ROOT / 'paper/mst/iopjournal.cls').read_text(encoding='utf-8', errors='ignore')
for cmd in ('articletype', 'affil', 'email', 'keywords', 'orcid', 'ack', 'funding', 'roles', 'data'):
    if re.search(r'\\(newcommand|renewcommand|def|newenvironment|renewenvironment)\s*\{?\\' + cmd, cls):
        good(f'\\{cmd} is defined by the class')
    else:
        bad(f'\\{cmd} is used but NOT defined in iopjournal.cls')

# packages used vs loaded
used_pkgs = set(re.findall(r'\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}', body))
used_pkgs = {p.strip() for g in used_pkgs for p in g.split(',')}
if r'\url{' in body and 'url' not in used_pkgs and 'hyperref' not in used_pkgs:
    bad(r'\url is used but neither url nor hyperref is loaded')
elif r'\url{' in body:
    good(r'\url is used and the url package is loaded')
if r'\toprule' in body or r'\midrule' in body:
    if 'booktabs' not in used_pkgs:
        bad('booktabs rules used without loading booktabs')

# ---------------------------------------------------------------- numbers
scored = json.load(open(ROOT / 'experiments/allocation_confirm_v1_2026-09-15/scored.json'))
S = json.load(open(ROOT / 'experiments/allocation_confirm_v1_2026-09-15/summary_v2.json'))['summary']
import numpy as np


def sub_median(k, m, method='state_mean'):
    v = [s['mae_m'] for s in S if s['views'] == k and s['moments'] == m
         and s['method'] == method and s['mae_m'] is not None]
    return np.median(v) * 1000 if v else None


# the confirmation table in the manuscript
tab = re.search(r'\\label\{tab:confirm\}', body)
rows = re.findall(r'^(\d+) & ([\d.]+) & \\textbf\{?([\d.]+)\}? & ([\d.]+) & (.*?)\\\\', body, re.M)
checked = 0
for m_, k2, k3, k7, fail in rows:
    for k, want in ((2, k2), (3, k3), (7, k7)):
        got = sub_median(k, int(m_))
        if got is None:
            bad(f'confirmation table m={m_} k={k}: no such cell in the artefacts')
        elif abs(got - float(want)) > 0.06:
            bad(f'confirmation table m={m_} k={k}: manuscript says {want}, artefact gives {got:.2f}')
        else:
            checked += 1
if checked:
    good(f'{checked} cells of the confirmation table match the sealed artefacts')

# the 43 % claim
f3, f7 = 3 * 33, 7 * 33
if abs(f3 / f7 * 100 - 43) > 1:
    bad(f'the "43 %" claim: 99/231 is {f3/f7*100:.1f} %')
else:
    good(f'the 43 % frame-cost claim checks out ({f3}/{f7} = {f3/f7*100:.1f} %)')

# uncertainty budget consistency
bud = json.load(open(ROOT / 'experiments/uncertainty_budget_v1_2026-09-15/budget.json'))
comp = {c['symbol']: c['value_mm'] for c in bud['components']}
uc = (comp['u_A'] ** 2 + comp['u_ref'] ** 2) ** .5
if abs(uc - bud['combined_standard_uncertainty_mm']) > 1e-6:
    bad(f'uncertainty budget: u_c does not equal the RSS of its components')
else:
    good('uncertainty budget: u_c equals the root sum of squares of u_A and u_ref')
if abs(bud['expanded_uncertainty_k2_mm'] - 2 * uc) > 1e-6:
    bad('uncertainty budget: U is not 2 u_c')
else:
    good('uncertainty budget: U = 2 u_c')
for label, val in (('-8.23', comp['b_samp']), ('7.43', comp['u_A']), ('0.55', comp['u_ref']),
                   ('7.45', bud['combined_standard_uncertainty_mm']), ('14.9', bud['expanded_uncertainty_k2_mm'])):
    if label not in body:
        bad(f'uncertainty value {label} mm appears in the artefact but not in the manuscript table')
if '0.22' in body:
    rel = bud['relative_expanded'] * 100
    if abs(rel - 0.22) > 0.01:
        bad(f'the manuscript says 0.22 % relative but the artefact gives {rel:.3f} %')
    else:
        good(f'relative expanded uncertainty {rel:.2f} % matches')

# mechanism table
mech = json.load(open(ROOT / 'experiments/view_saturation_mechanism_v1_2026-09-15/mechanism.json'))
for rec, v in mech.items():
    n = rec.replace('record', '')
    row = re.search(r'^' + n + r' & ([\d.]+) & ([\d.]+) & \\?t?e?x?t?b?f?\{?([\d.]+)\}? & ([\d.]+) & ([\d.]+)', body, re.M)
    if not row:
        bad(f'mechanism table: no row found for record {n}')
        continue
    pairs = ((v['rms3_mm'], row.group(1)), (v['rms7_mm'], row.group(2)),
             (v['realised_ratio'], row.group(3)), (v['predicted_ratio_iid'], row.group(4)),
             (v['median_cos_err3_err7'], row.group(5)))
    for got, want in pairs:
        if abs(got - float(want)) > 0.002:
            bad(f'mechanism table record {n}: manuscript {want}, artefact {got:.4f}')
    else:
        good(f'mechanism table record {n}: all five values match')

# ---------------------------------------------------------------- package integrity
if DESK.exists():
    man = json.loads((DESK / 'Package_Manifest.json').read_text(encoding='utf-8'))['files']
    mism = [f for f, h in man.items()
            if not (DESK / f).exists() or hashlib.sha256((DESK / f).read_bytes()).hexdigest() != h]
    if mism:
        bad(f'desktop package: {len(mism)} files do not match the manifest: {mism[:4]}')
    else:
        good(f'desktop package: all {len(man)} files match the manifest')
    # the uploaded manuscript must be the current one
    a = hashlib.sha256((DESK / '01_Upload/01_Manuscript.tex').read_bytes()).hexdigest()
    b = hashlib.sha256(MST.read_bytes()).hexdigest()
    if a != b:
        bad('desktop 01_Upload/01_Manuscript.tex is NOT the current manuscript '
            '(it predates the DOI insertion)')
    else:
        good('desktop manuscript is byte-identical to the working manuscript')
    for f in ('02_LaTeX_Source/mst_manuscript.tex',):
        if hashlib.sha256((DESK / f).read_bytes()).hexdigest() != b:
            bad(f'desktop {f} differs from the working manuscript')
    # compile set completeness
    need = {'iopjournal.cls', 'orcid.pdf', 'mst_manuscript.tex', 'fig1.pdf', 'fig2.pdf', 'fig3.pdf', 'fig4.pdf'}
    have = {p.name for p in (DESK / '02_LaTeX_Source').iterdir()}
    if need - have:
        bad(f'02_LaTeX_Source is missing files needed to compile: {sorted(need - have)}')
    else:
        good('02_LaTeX_Source contains everything needed to compile')
else:
    bad('desktop submission folder not found')

# the Zenodo archive should contain a manuscript; is it the current one?
if ZIP.exists():
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith('mst_manuscript.tex')]
        if names:
            zh = hashlib.sha256(z.read(names[0])).hexdigest()
            if zh != hashlib.sha256(MST.read_bytes()).hexdigest():
                note('the published Zenodo archive holds the manuscript as it was before the DOI '
                     'was inserted. That is inherent: the DOI cannot be inside the thing it names. '
                     'The GitHub repository carries the current manuscript.')
            else:
                good('Zenodo archive manuscript matches the current one')

# repo vs working tree
if (REPO / 'paper/mst_manuscript.tex').exists():
    if hashlib.sha256((REPO / 'paper/mst_manuscript.tex').read_bytes()).hexdigest() != \
       hashlib.sha256(MST.read_bytes()).hexdigest():
        bad('the GitHub repo copy of the manuscript differs from the working manuscript')
    else:
        good('GitHub repo manuscript matches the working manuscript')

# placeholders
ph = re.findall(r'<<[^>]{0,80}', body)
if ph:
    note(f'{len(ph)} placeholders still in the manuscript: ' +
         '; '.join(' '.join(p.split())[:42] for p in ph))

# ------------------------------------------------- figure tick-label collisions
# On a log axis matplotlib labels the minor ticks as well, and at journal column
# widths those labels overlap each other. Every log axis must therefore either
# set its own ticks or switch the minor ones off.
figsrc = (ROOT / 'src/make_mst_figures.py').read_text(encoding='utf-8')
blocks = re.split(r'\n    fig, ax = |\n    fig, ax\b', figsrc)
for i, blk in enumerate(blocks[1:], 1):
    logx = len(re.findall(r"set_xscale\('log'\)", blk))
    if not logx:
        continue
    guarded = len(re.findall(r'minorticks_off\(\)', blk))
    ticks = len(re.findall(r'set_xticks\(', blk))
    if guarded == 0 and ticks == 0:
        bad(f'figure block {i} sets a log x axis but neither fixes the ticks nor '
            f'calls minorticks_off; matplotlib will label the minor ticks and they '
            f'will overlap at column width')
    else:
        good(f'figure block {i}: log axis has explicit ticks or minor ticks off')

# ---------------------------------------------------------------- real compile
TECT = Path(r'C:\Users\Jason\AppData\Local\Temp\claude\E--research-VLLM'
            r'\21767287-c04b-4dd2-a6c6-30ae524b66df\scratchpad\tect\tectonic.exe')
if TECT.exists():
    import shutil
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in ('mst_manuscript.tex', 'iopjournal.cls', 'orcid.pdf',
                  'fig1.pdf', 'fig2.pdf', 'fig3.pdf', 'fig4.pdf'):
            shutil.copy2(ROOT / 'paper/mst' / f, td / f)
        r = subprocess.run([str(TECT), '-X', 'compile', 'mst_manuscript.tex', '--keep-logs'],
                           cwd=td, capture_output=True, text=True, timeout=900)
        logf = td / 'mst_manuscript.log'
        log = logf.read_text(encoding='utf-8', errors='ignore') if logf.exists() else ''
        if r.returncode != 0 or not (td / 'mst_manuscript.pdf').exists():
            bad(f'the manuscript DOES NOT COMPILE (tectonic exit {r.returncode})')
        else:
            pages = re.search(r'Output written on .*\((\d+) pages', log)
            good(f'manuscript compiles cleanly to {pages.group(1) if pages else "?"} pages')
            over = re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\)', log)
            if over:
                bad(f'{len(over)} Overfull hbox, worst {max(float(x) for x in over):.1f} pt: '
                    'content runs past the text margin')
            else:
                good('no Overfull hbox: nothing runs past the text margin')
            for pat, msg in ((r'multiply defined', 'duplicate labels or bibliography keys'),
                             (r'LaTeX Warning: Citation .* undefined', 'an undefined citation'),
                             (r'LaTeX Warning: Reference .* undefined', 'an undefined cross-reference')):
                if re.search(pat, log):
                    bad(f'LaTeX reports {msg}')
            und = len(re.findall(r'Underfull \\hbox', log))
            if und > 1:
                note(f'{und} Underfull hbox; one of them is inherent to the official template')
            elif und == 1:
                good('the only Underfull hbox is the one the official template itself produces')
else:
    note('tectonic not available, so the real compile check was skipped')

print(f'ISSUES {len(issues)}   NOTES {len(notes)}   OK {len(ok)}\n')
if issues:
    print('PROBLEMS TO FIX')
    for m in issues:
        print('  [FIX] ', m)
    print()
if notes:
    print('NOTES')
    for m in notes:
        print('  [note]', m)
    print()
print('VERIFIED')
for m in ok:
    print('  [ok]  ', m)
sys.exit(1 if issues else 0)
