"""Assemble the Measurement (Elsevier) submission package into one folder.

paper/measurement/submission_package/
  01_Manuscript_anonymized.pdf        main file, double-anonymized
  02_Title_page.docx / .pdf           authors, affiliations, CRediT, declarations (NOT anonymized)
  03_Highlights.docx                  3-5 bullets, separate editable file with "Highlights" in the name
  04_Supplementary_material.pdf       anonymized
  05_Cover_letter.docx
  06_Declaration_of_competing_interest.docx
  07_LaTeX_source_anonymized.zip      manuscript + supplement source and figures (compiles on its own)
  08_Form_fields_to_paste.txt         title, abstract, keywords, reviewers, statements for the web form
  00_CHECKLIST.md

Usage: python src/build_measurement_package.py
"""
from __future__ import annotations

import re
import shutil
import sys
import subprocess
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "paper" / "measurement"
OUT = M / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "submission_package")
BS = chr(92)
TECT = Path(r"C:/Users/Jason/AppData/Local/Temp/claude/E--research-VLLM/21767287-c04b-4dd2-a6c6-30ae524b66df/scratchpad/tect/tectonic.exe")


def doc_from_paragraphs(path: Path, title: str, paragraphs, bullets=False):
    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"
    d.styles["Normal"].font.size = Pt(12)
    d.add_heading(title, level=1)
    for p in paragraphs:
        if bullets:
            d.add_paragraph(p, style="List Bullet")
        else:
            d.add_paragraph(p)
    d.save(str(path))


def tex_to_text(s: str) -> str:
    s = s.replace("--", "\u2013").replace(BS + "&", "&").replace(BS + ",", " ").replace("~", " ")
    s = re.sub(re.escape(BS) + r"(textbf|emph|url|texttt)\{([^}]*)\}", r"\2", s)
    return " ".join(s.split())


def section(tex: str, name: str) -> str:
    i = tex.index(BS + "section*{" + name + "}") + len(BS + "section*{" + name + "}")
    j = tex.find(BS + "section*{", i)
    j = tex.index(BS + "end{document}") if j < 0 else j
    return tex[i:j].strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.iterdir():          # empty the folder but keep it (it may be open in a shell or explorer)
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    ms = (M / "measurement_manuscript.tex").read_text(encoding="utf-8")
    tp = (M / "title_page.tex").read_text(encoding="utf-8")

    shutil.copy2(M / "measurement_manuscript.pdf", OUT / "01_Manuscript_anonymized.pdf")
    shutil.copy2(M / "title_page.pdf", OUT / "02_Title_page.pdf")
    shutil.copy2(M / "supplementary_material.pdf", OUT / "04_Supplementary_material.pdf")

    # title page as Word
    authors = re.findall(re.escape(BS) + r"author\[([^\]]+)\]\{([^}\\]+)", tp)
    corr = re.search(re.escape(BS) + r"author\[[^\]]+\]\{([^}\\]+)" + re.escape(BS) + r"corref", tp).group(1).strip()
    corr_mail = re.search(re.escape(BS) + r"ead\{([^}]*)\}", tp).group(1).strip()
    affs = dict(re.findall(re.escape(BS) + r"affiliation\[([^\]]+)\]\{(.*)\}\s*$", tp, flags=re.M))
    order = []
    for key, _ in authors:
        if key not in order:
            order.append(key)
    letters = {k: "abcdefghij"[i] for i, k in enumerate(order)}
    def aff_text(a):
        parts = dict(re.findall(r"(\w+)=\{([^}]*)\}", a))
        return ", ".join(parts[k] for k in ("organization", "addressline", "city", "postcode", "state", "country") if k in parts)
    title = re.search(re.escape(BS) + r"title\{([^}]*)\}", tp).group(1)
    paras = [title, ", ".join("%s (%s)%s" % (n.strip(), letters[k], " *" if n.strip() == corr else "") for k, n in authors)]
    paras += ["(%s) %s" % (letters[k], aff_text(affs[k])) for k in order]
    paras += ["* Corresponding author: %s, %s" % (corr, corr_mail),
              "Short title: Frame budget: views or instants",
              tex_to_text(re.search(r"ORCID:\}(.*)", tp).group(1))]
    for name in ("Declaration of competing interest", "Funding", "CRediT authorship contribution statement",
                 "Data availability", "Preprint and prior posting"):
        paras += [name.upper(), tex_to_text(section(tp, name))]
    doc_from_paragraphs(OUT / "02_Title_page.docx", "Title page", paras)

    hl = [l.lstrip("\u2022 ").strip() for l in (M / "highlights.txt").read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    assert 3 <= len(hl) <= 5 and all(len(h) <= 85 for h in hl)
    doc_from_paragraphs(OUT / "03_Highlights.docx", "Highlights", hl, bullets=True)

    cl = (M / "cover_letter.txt").read_text(encoding="utf-8").replace(
        "[Corresponding author name, on behalf of all authors -- see title page]",
        "%s (corresponding author), on behalf of all authors\nCollege of Computer Science and Technology, Qingdao University, Qingdao 266071, China\n%s" % (corr, corr_mail))
    (OUT / "05_Cover_letter.txt").write_text(cl, encoding="utf-8")
    doc_from_paragraphs(OUT / "05_Cover_letter.docx", "Cover letter", [p for p in cl.split("\n\n") if p.strip()])

    doc_from_paragraphs(OUT / "06_Declaration_of_competing_interest.docx", "Declaration of competing interest",
                        [tex_to_text(section(tp, "Declaration of competing interest"))])

    # anonymized LaTeX source, verified to compile on its own
    zpath = OUT / "07_LaTeX_source_anonymized.zip"
    figs = re.findall(re.escape(BS) + r"includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", ms)
    files = ["measurement_manuscript.tex", "refs_elsevier.tex", "supplementary_material.tex"] + figs
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(M / f, f)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        for f in ("measurement_manuscript.tex", "supplementary_material.tex"):
            r = subprocess.run([str(TECT), f], cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0 or not (Path(tmp) / f.replace(".tex", ".pdf")).exists():
                raise SystemExit("source zip does not compile on its own: %s\n%s" % (f, r.stderr[-800:]))
    print("source zip compiles on its own:", ", ".join(files))

    ab = ms[ms.index(BS + "begin{abstract}") + 16: ms.index(BS + "end{abstract}")]
    ab_txt = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace(BS + "ast", "*").replace("^", "").replace(BS, ""), ab)
    ab_txt = " ".join(ab_txt.replace("--", "\u2013").replace(BS + ",", " ").split())
    kw = [k.strip() for k in ms[ms.index(BS + "begin{keyword}") + 15: ms.index(BS + "end{keyword}")].split(BS + "sep")]
    ai = re.search(r"During the preparation of this work.*?publication\.", ms, flags=re.S).group(0)
    form = ["ARTICLE TYPE: Research Paper (full-length article)", "",
            "TITLE: " + re.search(re.escape(BS) + r"title\{([^}]*)\}", ms).group(1), "",
            "SHORT TITLE: Frame budget: views or instants", "",
            "ABSTRACT (%d words):" % len(re.sub(r"\S*\*\S*", "x", ab_txt).split()), ab_txt, "",
            "KEYWORDS (%d): " % len(kw) + "; ".join(kw), "",
            "HIGHLIGHTS:"] + ["- " + h for h in hl] + ["",
            "FUNDING: " + tex_to_text(section(tp, "Funding")), "",
            "DATA AVAILABILITY (statement for the submission form): The code and data are publicly available: "
            "https://github.com/Jun-Jason-Ji/views-or-instants and https://doi.org/10.5281/zenodo.22985306 "
            "(the manuscript itself withholds these links for double-anonymized review).", "",
            "GENERATIVE AI DECLARATION (already in the manuscript, before the references):", " ".join(ai.split()), "",
            "SUGGESTED REVIEWERS:",
            # internal notes ("Not suggested: ...") stay in suggested_reviewers.txt, not in the form
            (M / "suggested_reviewers.txt").read_text(encoding="utf-8").split("\nNot suggested:")[0].rstrip() + "\n"]
    (OUT / "08_Form_fields_to_paste.txt").write_text("\n".join(form), encoding="utf-8")
    print("package written to", OUT)
    for p in sorted(OUT.iterdir()):
        print("  %-42s %8.1f kB" % (p.name, p.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
