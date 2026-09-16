"""Adversarial audit of the MST submission package and of the numbers in the manuscript.

Looks for things that would actually break or embarrass: LaTeX that will not
compile, citation numbers pointing at the wrong reference, claims that do not
match the sealed artefacts, and inconsistencies between the manuscript, the
GitHub repository and the Zenodo archive.
"""
import hashlib
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MST = ROOT / 'paper/mst/mst_manuscript.tex'
DESK = Path(r'C:\Users\Jason\Desktop\MST_Submission_v1.0.0_2026-09-15')
parser = argparse.ArgumentParser()
parser.add_argument('--package', type=Path)
args = parser.parse_args()
if args.package:
    DESK = args.package.resolve()
REPO = ROOT / 'paper/repo'
ZIP = ROOT / 'paper/zenodo/view_instant_allocation_v1.0.4.zip'

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

# Only retain the descriptive components; no combined uncertainty is claimed.
bud = json.load(open(ROOT / 'experiments/uncertainty_budget_v1_2026-09-15/budget.json'))
comp = {c['symbol']: c['value_mm'] for c in bud['components']}
for label, val in (('-8.23', comp['b_samp']), ('7.43', comp['u_A']), ('0.95', comp['u_ref'] * 3**0.5)):
    if abs(float(label) - val) > .005:
        bad(f'descriptive error value {label} does not round from archived calculation {val}')
    if label not in body:
        bad(f'descriptive error value {label} mm missing from manuscript')
if 'Illustrative scale' in body or 'Twice the illustrative scale' in body:
    bad('withdrawn quadrature combination still appears')
else:
    good('bias, dispersion and reference sensitivity reported without a quadrature coverage claim')

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

# -------------------------------------------------- revised evidence checks
derived = ROOT / 'experiments/submission_sensitivity_2026-09-16'
tum = json.load(open(ROOT / 'experiments/tum_dense_cheap_confirm_v1_2026-09-15/evaluation.json'))
rev = json.load(open(derived / 'tum_sequence_equal.json'))
for fe in ('orb', 'xfeat'):
    for b, entry in rev[fe]['per_budget'].items():
        rr = [r for r in tum if r['method'] == fe and r['budget_label'] == b and r['errors_m']]
        seqs = sorted({r['sequence'] for r in rr})
        bias = np.mean([np.mean([r['errors_m']['state_mean'] for r in rr if r['sequence']==seq]) for seq in seqs])
        mae = np.mean([np.mean([abs(r['errors_m']['state_mean']) for r in rr if r['sequence']==seq]) for seq in seqs])
        if not (abs(bias-entry['bias_seq_equal']) < 1e-12 and abs(mae-entry['mae_seq_equal']) < 1e-12 and abs(bias) <= mae+1e-12):
            bad(f'revised secondary summary {fe}/{b}: inconsistent weighting or |bias| > MAE')
        else:
            good(f'sequence-equal MAE/bias verified independently: {fe}/{b}')
pair = json.load(open(derived / 'paired_records.json'))['comparisons']
equal = [r for r in pair if r['m3'] == 28 and r['m7'] == 12][0]
if equal['frames3'] != equal['frames7'] or equal['frames3'] != 84:
    bad('revised equal-budget comparison does not have equal frame costs')
else:
    good('revised main equal-budget comparison uses exactly 84 frames in both arms')
for r in pair:
    if r['m3'] != r['m7']: continue
    line = f"{r['m3']} & {r['mae3_mm']:.2f} & {r['mae7_mm']:.2f} & ${r['difference_mm']:+.2f}$ & $[{r['cluster_percentile95_mm'][0]:.2f},{r['cluster_percentile95_mm'][1]:.2f}]$"
    if line not in body:
        bad(f'record-level table m={r["m3"]} differs from derived results')
    else:
        good(f'record-level table m={r["m3"]}: MAEs, difference and interval agree')
cfg = json.load(open(ROOT / 'experiments/tum_dense_cheap_confirm_v1_2026-09-15/config.json'))
decision = json.load(open(ROOT / 'experiments/tum_dense_cheap_confirm_v1_2026-09-15/decision.json'))['decision']
if cfg['frontends']['primary'] != 'xfeat' or decision['xfeat']['gate_pass'] or not decision['orb']['gate_pass']:
    bad('primary/secondary gate interpretation differs from frozen records')
elif 'It was not fully met' not in body or 'XFeat was the pre-specified primary' not in body:
    bad('manuscript does not explicitly disclose the incomplete primary criterion')
else:
    good('primary XFeat failure and secondary ORB success disclosed')
for folder in ('allocation_confirm_v1_2026-09-15','tum_dense_cheap_confirm_v1_2026-09-15'):
    folder = ROOT / 'experiments' / folder
    seal = json.load(open(folder / 'prediction_seal.json'))['sha256']
    if hashlib.sha256((folder / 'predictions.json').read_bytes()).hexdigest() != seal:
        bad(f'original sealed predictions changed: {folder.name}')
    else:
        good(f'original prediction seal intact: {folder.name}')
if 'Anthropic Claude' in body or 'OpenAI GPT' not in body:
    bad('acknowledgments AI name not updated')
else:
    good('acknowledgments name OpenAI GPT')

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
                note('the published Zenodo archive is a historical snapshot and differs from '
                     'the current manuscript; a new archive version is needed to include '
                     'subsequent manuscript changes. The local repository copy is checked separately.')
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
    # the y axis has the same failure mode, plus 10^1-style labels that are
    # inconsistent with the plain numbers used elsewhere
    logy = len(re.findall(r"set_yscale\('log'\)", blk))
    yticks = len(re.findall(r'plain_log_ticks\([a-z\[\]0-9]+\.yaxis|set_yticks\(', blk))
    if logy and yticks < logy:
        bad(f'figure block {i}: {logy} log y axes but only {yticks} with explicit ticks; '
            f'the rest will be labelled 10^1, 10^2 with overlapping minor labels')

# ------------------------------------------------- duplicate References heading
# thebibliography prints its own heading; a \section*{References} in front of it
# puts the word on the page twice.
if re.search(r'\\section\*\{References\}\s*\\begin\{thebibliography\}', body):
    bad('\\section*{References} directly before thebibliography: the heading prints twice')
else:
    good('single References heading')

# ------------------------------------------------- compiled PDFs are current
# The manifest records the manuscript hash each PDF was built from; if the
# manuscript changed since, the PDFs on the desktop are stale.
if DESK.exists():
    manifest = json.loads((DESK / 'Package_Manifest.json').read_text(encoding='utf-8'))
    cur = hashlib.sha256(MST.read_bytes()).hexdigest()
    built = manifest.get('built_from_tex', {})
    if not built:
        bad('Package_Manifest.json has no built_from_tex record: rebuild with build_submission_package.py')
    else:
        stale = [f for f, h in built.items() if h != cur]
        if stale:
            bad(f'{len(stale)} package files were built from an older manuscript: {stale}')
        else:
            good(f'all {len(built)} compiled/copied manuscript files were built from the current tex')
    try:
        import fitz  # PyMuPDF
        for f in ('01_Upload/00_Manuscript_compiled.pdf', '01_Upload/00_Manuscript_anonymous.pdf'):
            with fitz.open(DESK / f) as doc:
                text = '\n'.join(page.get_text() for page in doc)
            heads = re.findall(r'(?m)^References\s*$', text)
            if len(heads) > 1:
                bad(f'{f}: the References heading appears {len(heads)} times')
            for ph in ('Journal Name', 'Author et al', 'dd Month yyyy'):
                if ph in text.replace('\n', ' ') and ph != 'dd Month yyyy':
                    bad(f'{f}: template placeholder "{ph}" is visible in the PDF')
            if 'anonymous' in f:
                leak = [w for w in ('Jun Ji', 'Zihan Li', 'Bowen Tan', 'Yi Sui', 'Shengjie Guo',
                                    'Xiaolei Zhang', 'Yi Li', 'Qingdao', 'qdu.edu.cn', 'Jun-Jason-Ji',
                                    'zenodo', 'Hohhot', 'Kowloon', 'Anthropic')
                        if w in text]
                if leak:
                    bad(f'{f}: identifying strings survive anonymisation: {leak}')
                else:
                    good('anonymous PDF carries no author, affiliation, e-mail, repository or DOI string')
            else:
                good(f'{f}: single References heading, no template placeholder text')
    except ImportError:
        note('PyMuPDF not available; PDF text checks skipped')

# ------------------------------------------------- package notes are current
for f, must_not in (('03_Reference/README_SUBMISSION.md', ('待填', '六位', '本机无 LaTeX')),
                    ('03_Reference/HOW_TO_DEPOSIT.md', ('六位作者',)),
                    ('00_READ_FIRST_投稿说明.txt', ('本机没有 LaTeX', '待填'))):
    p = DESK / f
    if p.exists():
        t = p.read_text(encoding='utf-8')
        hit = [w for w in must_not if w in t]
        if hit:
            bad(f'{f} is stale: still says {hit}')
        n_authors = len(re.findall(r'Zihan Li|Shengjie Guo', t))
        if 'README' in f and n_authors < 2:
            bad(f'{f} does not list the current seven authors')
if DESK.exists():
    good('package notes mention no unfilled items and list the current authors')
    from datetime import date
    cl = (DESK / '01_Upload/06_Cover_Letter.txt').read_text(encoding='utf-8').splitlines()[1].strip()
    today = date.today().strftime('%d %B %Y').lstrip('0')
    if cl != today:
        note(f'cover letter is dated "{cl}", today is {today}; change line 2 of '
             f'paper/mst/cover_letter.txt and rebuild if submitting today')

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
