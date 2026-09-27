"""Look up every \\bibitem of a manuscript on Crossref (title search) and save the
metadata with a title-similarity score, for converting the reference list to a
journal style.  Matches below the similarity threshold must be checked by hand.

Usage: python src/crossref_lookup_refs.py paper/tim/tim_manuscript.tex paper/measurement/crossref_refs.json
"""
import difflib
import json
import re
import sys
import urllib.parse
import urllib.request

BS = chr(92)


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    s = open(src, encoding="utf-8").read()
    bib = s[s.index(BS + "begin{thebibliography}"):s.index(BS + "end{thebibliography}")]
    items = re.split(re.escape(BS) + r"bibitem\{", bib)[1:]
    out = []
    for it in items:
        key, body = it.split("}", 1)
        body = " ".join(body.split())
        m = re.search(r"(?:19|20)\d\d\s+(.*?)(?:" + re.escape(BS) + r"textit|$)", body)
        title = (m.group(1) if m else body[:150]).strip()
        rec = {}
        try:
            u = ("https://api.crossref.org/works?rows=1&mailto=junjiqd@gmail.com&query.bibliographic="
                 + urllib.parse.quote(title[:200]))
            d = json.load(urllib.request.urlopen(u, timeout=30))["message"]["items"][0]
            t = (d.get("title") or [""])[0]
            rec = dict(doi=d.get("DOI"), title=t,
                       authors=[(a.get("given", ""), a.get("family", a.get("name", ""))) for a in d.get("author", [])],
                       journal=(d.get("container-title") or [""])[0],
                       volume=d.get("volume"), issue=d.get("issue"), page=d.get("page"),
                       article=d.get("article-number"),
                       year=(d.get("issued", {}).get("date-parts") or [[None]])[0][0], type=d.get("type"),
                       sim=round(difflib.SequenceMatcher(None, t.lower(), title.lower()).ratio(), 2))
        except Exception as e:  # noqa: BLE001
            rec = dict(error=str(e))
        out.append(dict(key=key, body=body, query_title=title, crossref=rec))
        print(key, rec.get("sim"), rec.get("doi"), (rec.get("title") or "")[:70], flush=True)
    open(dst, "w", encoding="utf-8").write(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
