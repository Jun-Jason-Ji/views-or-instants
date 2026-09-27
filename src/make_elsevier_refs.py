"""Format the reference list in the Elsevier numbered style from the Crossref
metadata saved by crossref_lookup_refs.py, with hand-checked overrides for the
entries Crossref matched to a preprint or cannot resolve.

Output: a LaTeX file of \\bibitem entries, in the order of the input list (the
manuscript must cite them in that order, or the list is re-ordered by first
citation with --order-from MANUSCRIPT.tex).

Usage: python src/make_elsevier_refs.py paper/measurement/crossref_refs.json \
           paper/measurement/refs_elsevier.tex [--order-from paper/measurement/measurement_manuscript.tex]
"""
import json
import re
import sys
import urllib.request

BS = chr(92)
# DOIs confirmed by hand on 2026-09-26 (Crossref title search had returned a preprint)
DOI_OVERRIDE = {"walker2026": "10.1016/j.jbiomech.2026.113522", "nakano2020": "10.3389/fspor.2020.00050"}
MANUAL = {
    "sarkka2023": "S. S" + BS + '"{a}rkk' + BS + '"{a}, L. Svensson, Bayesian Filtering and Smoothing, second ed., '
                  "Cambridge University Press, Cambridge, 2023. " + BS + "url{https://doi.org/10.1017/9781108917407}.",
    "muller2025": "A. Muller, A. Na" + BS + '"{i}m, R. Dumas, T. Robert, Benchmarking dataset for markerless motion '
                  "capture analysis (version 2.0), Recherche Data Gouv, 2025. " + BS + "url{https://doi.org/10.57745/LQI2MJ}.",
    "archive": "[Reference withheld for double-anonymized review.]",
    "jcgm200": "JCGM 200:2012, International vocabulary of metrology -- Basic and general concepts and associated "
               "terms (VIM), third ed., Joint Committee for Guides in Metrology, 2012. " + BS + "url{https://doi.org/10.59161/JCGM200-2012}.",
    "jcgm100": "JCGM 100:2008, Evaluation of measurement data -- Guide to the expression of uncertainty in measurement "
               "(GUM), Joint Committee for Guides in Metrology, 2008. " + BS + "url{https://doi.org/10.59161/JCGM100-2008E}.",
    "jcgm101": "JCGM 101:2008, Evaluation of measurement data -- Supplement 1 to the Guide to the expression of "
               "uncertainty in measurement -- Propagation of distributions using a Monte Carlo method, Joint Committee "
               "for Guides in Metrology, 2008. " + BS + "url{https://doi.org/10.59161/JCGM101-2008}.",
}
EXTRA = {"primaryrig": " MCalib dataset: " + BS + "url{https://koonyook.github.io/MCalib/}."}

LATEX = {"&": BS + "&", "%": BS + "%", "_": BS + "_", "#": BS + "#",
         "–": "--", "—": "---", "’": "'", "‘": "`", " ": " "}
ACCENT = {"ä": '"{a}', "ö": '"{o}', "ü": '"{u}', "é": "'{e}", "è": "`{e}",
          "ï": '"{i}', "Ś": "'{S}", "ń": "'{n}", "ć": "'{c}", "ł": "{l}",
          "ó": "'{o}", "á": "'{a}", "í": "'{i}", "ż": ".{z}", "ę": "k{e}"}


def tex(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    for a, b in LATEX.items():
        s = s.replace(a, b)
    for a, b in ACCENT.items():
        s = s.replace(a, BS + b)
    return " ".join(s.split())


def initials(given):
    parts = re.split(r"[\s]+", given.strip())
    out = []
    for p in parts:
        if not p:
            continue
        sub = [q for q in p.split("-") if q]
        out.append("-".join(q[0] + "." for q in sub))
    return "".join(out)


CACHE = {}


def fetch(doi):
    """Crossref record for a DOI, cached on disk next to the output, with retries."""
    import time
    if doi in CACHE:
        return CACHE[doi]
    for attempt in range(5):
        try:
            CACHE[doi] = json.load(urllib.request.urlopen("https://api.crossref.org/works/" + doi, timeout=30))["message"]
            return CACHE[doi]
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def fmt(meta):
    auth = meta.get("author", [])
    names = [(initials(a.get("given", "")) + " " + a.get("family", "")).strip() if a.get("family")
             else a.get("name", "") for a in auth]
    if len(names) > 10:
        names = names[:10] + ["et al."]
    title = (meta.get("title") or [""])[0]
    jour = (meta.get("container-title") or [""])[0]
    # the year of the printed volume, not of online-first publication
    year = None
    for k in ("published-print", "journal-issue", "published", "issued"):
        dp = meta.get(k, {})
        if k == "journal-issue":
            dp = dp.get("published-print", {}) if isinstance(dp, dict) else {}
        if dp.get("date-parts") and dp["date-parts"][0] and dp["date-parts"][0][0]:
            year = dp["date-parts"][0][0]
            break
    vol = meta.get("volume")
    loc = (meta.get("article-number") or meta.get("page") or "").replace("-", "--")
    s = ", ".join(tex(n) for n in names) + ", " + tex(title) + ", " + tex(jour)
    s += (" " + str(vol) if vol else "") + " (" + str(year) + ")" + ((" " + tex(loc)) if loc else "") + "."
    s += " " + BS + "url{https://doi.org/" + meta["DOI"] + "}."
    return s


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    refs = json.load(open(src, encoding="utf-8"))
    order = [r["key"] for r in refs]
    if "--order-from" in sys.argv:
        ms = open(sys.argv[sys.argv.index("--order-from") + 1], encoding="utf-8").read()
        body = ms.split(BS + "begin{thebibliography}")[0]
        seen = []
        for m in re.finditer(re.escape(BS) + r"cite(?:\[[^\]]*\])?\{([^}]*)\}", body):
            for k in m.group(1).split(","):
                k = k.strip()
                if k and k not in seen:
                    seen.append(k)
        missing = [k for k in seen if k not in order]
        if missing:
            raise SystemExit("cited but unknown: %s" % missing)
        order = seen
    by = {r["key"]: r for r in refs}
    cache_path = dst + ".crossref_cache.json"
    try:
        CACHE.update(json.load(open(cache_path, encoding="utf-8")))
    except (OSError, ValueError):
        pass
    lines = [BS + "begin{thebibliography}{%d}" % len(order), ""]
    for k in order:
        if k in MANUAL:
            s = MANUAL[k]
        else:
            doi = DOI_OVERRIDE.get(k) or by[k]["crossref"]["doi"]
            s = fmt(fetch(doi)) + EXTRA.get(k, "")
        lines += [BS + "bibitem{%s} %s" % (k, s), ""]
        print(k, "|", s[:110], flush=True)
    lines.append(BS + "end{thebibliography}")
    open(cache_path, "w", encoding="utf-8").write(json.dumps(CACHE))
    open(dst, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
