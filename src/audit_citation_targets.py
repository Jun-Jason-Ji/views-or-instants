"""Check that each hardcoded [n] in the text points at the reference it claims.

The manuscript numbers its citations by hand, so a reference inserted or moved
would silently shift every later number. This pairs the author name written next
to each citation with the author name in the bibliography entry it points to.
"""
import re
import sys
from pathlib import Path

MST = Path(__file__).resolve().parents[1] / 'paper/mst/mst_manuscript.tex'
src = re.sub(r'(?m)^%.*$', '', MST.read_text(encoding='utf-8'))

bibpart = src.split(r'\begin{thebibliography}')[1]
entries = re.findall(r'\\bibitem\{([^}]*)\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})', bibpart, re.S)
surnames = {}
for i, (key, text) in enumerate(entries, 1):
    t = ' '.join(text.split())
    m = re.match(r'([A-Z][A-Za-z\'\-]+)', t)
    surnames[i] = (m.group(1) if m else '?', key, t[:60])

bad, good = [], []
# a citation is "named" when an author surname appears within 60 chars before it
for m in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', src):
    if m.start() > src.index(r'\begin{thebibliography}'):
        continue
    nums = [int(x) for x in re.split(r',\s*', m.group(1))]
    before = ' '.join(src[max(0, m.start() - 90):m.start()].split())
    nm = re.findall(r'([A-Z][a-z]+)(?:\s+\\textit\{et al\}|\s+and\s+[A-Z][a-z]+)?\s*$', before)
    if not nm:
        continue
    claimed = nm[-1]
    target = surnames.get(nums[0], ('?', '', ''))[0]
    if claimed.lower() == target.lower():
        good.append(f'[{nums[0]}] near "{claimed}" -> {target}')
    else:
        bad.append(f'[{nums[0]}] is written next to "{claimed}" but entry {nums[0]} is '
                   f'"{target}" ({surnames.get(nums[0], ("", "", ""))[2]})')

print('bibliography order:')
for i, (sn, key, t) in surnames.items():
    print(f'  [{i:2d}] {key:<20} {t}')
print()
print(f'named citations checked: {len(good) + len(bad)}   mismatches: {len(bad)}')
for b in bad:
    print('  [FIX]', b)
for g in good:
    print('  [ok] ', g)
sys.exit(1 if bad else 0)
