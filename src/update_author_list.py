"""Propagate the current author list from the manuscript to every other artefact.

Single source of truth is the list below. Run after any change to authorship so
that the cover letter, the citation file, the Zenodo metadata and the GitHub
repository cannot drift apart from the manuscript.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# order, English name, Chinese name, affiliation key, ORCID or None, corresponding
AUTHORS = [
    ('Jun Ji',       '纪军',   'qdu',   '0000-0003-3194-2183', False),
    ('Zihan Li',     '李梓晗', 'maths', None,                  False),
    ('Bowen Tan',    '谭博文', 'hkust', '0009-0007-0554-9261', False),
    ('Yi Sui',       '隋毅',   'qdu',   '0009-0001-8081-5183', False),
    ('Shengjie Guo', '郭圣杰', 'imau',  '0009-0004-6852-4836', False),
    ('Xiaolei Zhang', '张晓蕾', 'qdu',  '0000-0002-0122-4554', False),
    ('Yi Li',        '李毅',   'qdu',   '0000-0002-4185-3152', True),
]
EMAIL = {'Jun Ji': 'junji@qdu.edu.cn', 'Zihan Li': '2072144@qq.com',
         'Bowen Tan': 'btanab@connect.ust.hk', 'Yi Sui': 'suiyi@qdu.edu.cn',
         'Shengjie Guo': 'guosj@emails.imau.edu.cn',
         'Xiaolei Zhang': 'zhangxiaolei@qdu.edu.cn', 'Yi Li': 'ly2005@qdu.edu.cn'}
AFFIL = {
    'qdu':   'College of Computer Science and Technology, Qingdao University, Qingdao, China',
    'maths': 'College of Mathematics and Statistics, <<UNIVERSITY NAME>>, China',
    'hkust': 'The Hong Kong University of Science and Technology, Hong Kong SAR, China',
    'imau':  'Inner Mongolia Agricultural University, Hohhot, China',
}


def zenodo_creators():
    out = []
    for name, _, aff, orcid, _ in AUTHORS:
        parts = name.rsplit(' ', 1)
        c = {'name': f'{parts[1]}, {parts[0]}', 'affiliation': AFFIL[aff]}
        if orcid:
            c['orcid'] = orcid
        out.append(c)
    return out


def cff_authors(indent='  '):
    lines = []
    for name, _, aff, orcid, _ in AUTHORS:
        given, family = name.rsplit(' ', 1)
        lines.append(f'{indent}- family-names: "{family}"')
        lines.append(f'{indent}  given-names: "{given}"')
        if orcid:
            lines.append(f'{indent}  orcid: "https://orcid.org/{orcid}"')
        lines.append(f'{indent}  affiliation: "{AFFIL[aff]}"')
    return '\n'.join(lines)


def main():
    # .zenodo.json in the repository
    p = ROOT / 'paper/repo/.zenodo.json'
    z = json.loads(p.read_text(encoding='utf-8'))
    z['creators'] = zenodo_creators()
    p.write_text(json.dumps(z, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'.zenodo.json: {len(z["creators"])} creators')

    # CITATION.cff: rewrite both author blocks
    p = ROOT / 'paper/repo/CITATION.cff'
    s = p.read_text(encoding='utf-8')
    s = re.sub(r'(?m)^authors:\n(?:  - .*\n|    .*\n)+', 'authors:\n' + cff_authors() + '\n', s, count=1)
    s = re.sub(r'(?m)^(  authors:\n)(?:    - .*\n|      .*\n)+',
               '  authors:\n' + cff_authors('    ') + '\n', s, count=1)
    p.write_text(s, encoding='utf-8')
    n = s.count('family-names:')
    print(f'CITATION.cff: {n} author entries across both blocks')

    # cover letter signature block
    p = ROOT / 'paper/mst/cover_letter.txt'
    s = p.read_text(encoding='utf-8')
    names = [a[0] for a in AUTHORS]
    s = re.sub(r'on behalf of .*$', 'on behalf of ' + ', '.join(names[:-1]) + ' and ' + names[-1],
               s, flags=re.S).rstrip() + '\n'
    p.write_text(s, encoding='utf-8')
    print(f'cover_letter.txt: signature lists {len(names)} authors')

    print('\ncurrent author list')
    for i, (name, cn, aff, orcid, corr) in enumerate(AUTHORS, 1):
        print(f'  {i}. {name:<14} {cn:<8} {EMAIL[name]:<28} '
              f'{orcid or "ORCID missing":<22} {"corresponding" if corr else ""}')


if __name__ == '__main__':
    main()
