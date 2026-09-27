"""Check the Measurement (Elsevier) submission package against the journal's rules.

Rules encoded from the Guide for Authors (Wayback copy of 2024-02-18, the live
page being behind a CAPTCHA) and the live Aims & Scope page (2026-09-26):
abstract <= 150 words; <= 6 keywords; 3-5 highlights of <= 85 characters;
double-anonymized manuscript; the generative-AI declaration with the prescribed
heading and wording, placed before the references; numbered references in order
of first citation; a data statement; VIM/GUM terminology; cover letter
explaining the fit to the scope.  Re-check the 2026 guide in a browser before
submitting.

Usage: python src/check_measurement_compliance.py [paper/measurement]
"""
import re
import sys
from pathlib import Path

BS = chr(92)
AI_HEAD = "Declaration of Generative AI and AI-assisted technologies in the writing process"
AI_TEXT = ("After using this tool/service, the author(s) reviewed and edited the content as needed "
           "and take(s) full responsibility for the content of the publication.")
IDENTITY = ["Tan", "Zihan", "Shengjie", "Sui", "Xiaolei", "Jun Ji", "Qingdao", "Hohhot", "HKUST",
            "junji", "github.com/Jun", "Jun-Jason", "zenodo.2278", "Hong Kong University"]
results = []


def chk(ok, msg, level="FAIL"):
    results.append(("ok" if ok else level, msg))


def words(tex):
    t = re.sub(r"\$[^$]*\$", "X", tex)
    t = re.sub(re.escape(BS) + r"[A-Za-z]+", "", t)
    return len(t.split())


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/measurement")
    ms = (d / "measurement_manuscript.tex").read_text(encoding="utf-8")
    body = ms.split(BS + "input{refs_elsevier.tex}")[0]
    refs = (d / "refs_elsevier.tex").read_text(encoding="utf-8")
    ab = ms[ms.index(BS + "begin{abstract}") + 16: ms.index(BS + "end{abstract}")]
    n = words(ab)
    chk(n <= 150, "abstract %d words (<= 150)" % n)
    chk("$$" not in ab and BS + "begin{equation}" not in ab, "abstract has no display formulae")
    kw = ms[ms.index(BS + "begin{keyword}") + 14: ms.index(BS + "end{keyword}")]
    nk = len([k for k in kw.split(BS + "sep") if k.strip()])
    chk(1 <= nk <= 6, "%d keywords (<= 6)" % nk)
    hl = [l.lstrip("•-* ").strip() for l in (d / "highlights.txt").read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    chk(3 <= len(hl) <= 5, "%d highlights (3-5)" % len(hl))
    for h in hl:
        chk(len(h) <= 85, "highlight %d chars <= 85: %s" % (len(h), h[:40]))
    # anonymization: manuscript, supplement, figures' text is not checked here
    for name in ("measurement_manuscript.tex", "supplementary_material.tex", "refs_elsevier.tex"):
        txt = (d / name).read_text(encoding="utf-8")
        hits = [w for w in IDENTITY if re.search(r"\b" + re.escape(w) + r"\b", txt)]
        chk(not hits, "%s carries no identifying string %s" % (name, hits if hits else ""))
    chk(BS + "author" not in body, "no \\author in the anonymized manuscript")
    pdf = d / "measurement_manuscript.pdf"
    if pdf.exists():
        try:
            import fitz  # PyMuPDF: extract the rendered text, which compressed streams hide from a byte search
            doc = fitz.open(str(pdf))
            txt = " ".join(p.get_text() for p in doc) + " " + " ".join(str(v) for v in doc.metadata.values() if v)
            hits = [w for w in IDENTITY if re.search(r"\b" + re.escape(w) + r"\b", txt)]
            chk(not hits, "compiled PDF text and metadata carry no identifying string %s" % (hits if hits else ""))
        except ImportError:
            chk(False, "PyMuPDF not installed: compiled PDF not checked for identifying strings", "warn")
    # AI declaration
    chk(AI_HEAD in ms, "AI declaration heading exactly as prescribed")
    chk(AI_TEXT in ms, "AI declaration closing sentence exactly as prescribed")
    chk("During the preparation of this work the author(s) used" in ms, "AI declaration opening as prescribed")
    chk(ms.find(AI_HEAD) < ms.find(BS + "input{refs_elsevier.tex}"), "AI declaration placed before the references")
    # references
    cited = []
    for m in re.finditer(re.escape(BS) + r"cite(?:\[[^\]]*\])?\{([^}]*)\}", body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in cited:
                cited.append(k)
    items = re.findall(re.escape(BS) + r"bibitem\{([^}]*)\}", refs)
    chk(set(cited) == set(items), "every citation has a reference and vice versa (%d)" % len(items))
    chk(cited == items, "references numbered in order of first citation")
    chk(all("doi.org" in b or "withheld" in b for b in re.split(re.escape(BS) + r"bibitem", refs)[1:]),
        "every reference carries a DOI (or is the withheld self-citation)", "warn")
    # content rules
    chk("Data availability" in ms, "data availability statement present")
    chk(BS + "linenumbers" in ms, "line numbers on for review", "warn")
    vim = len(re.findall(r"VIM|GUM|Type~A|Type~B|expanded uncertainty|systematic error", ms))
    chk(vim >= 8, "VIM/GUM vocabulary used (%d occurrences)" % vim)
    chk(not re.search(re.escape(BS) + r"(todo|dev)\{", ms), "no \\todo or \\dev markers")
    for f in re.findall(re.escape(BS) + r"includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", ms):
        chk((d / f).exists(), "figure file present: %s" % f)
    cl = (d / "cover_letter.txt").read_text(encoding="utf-8")
    chk("scope" in cl.lower() and "VIM" in cl and "GUM" in cl, "cover letter explains fit to scope in metrological terms")
    chk("not under consideration elsewhere" in cl, "cover letter states exclusivity")
    tp = (d / "title_page.tex").read_text(encoding="utf-8")
    todo = tp.count("[TO CONFIRM") + tp.count("[ZENODO")
    chk(todo == 0, "title page complete (%d item(s) still to confirm by the authors)" % todo, "warn")
    fails = sum(1 for r in results if r[0] == "FAIL")
    warns = sum(1 for r in results if r[0] == "warn")
    print("Measurement compliance: %d pass, %d FAIL, %d advisory" % (len(results) - fails - warns, fails, warns))
    for r in results:
        print("  %-5s %s" % r)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
