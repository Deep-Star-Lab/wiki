"""İçerik ve veri kontrolleri.  Çalıştır:  python3 tests/test_content.py"""
import datetime as dt
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dsl = load("dsl", "hooks/dsl.py")
fetch = load("fetch_repos", "scripts/fetch_repos.py")

# 1) Gerçek veri dosyaları geçerli mi?
dsl.load_data()
glossary = dsl.load_glossary()
dsl.load_reading()
print(f"veri geçerli: {len(glossary)} terim")

# 2) Her Türkçe sayfanın İngilizce karşılığı var mı? (katkı kuralı)
docs = ROOT / "docs"
missing = [str(p.relative_to(docs)) for p in sorted(docs.rglob("*.md"))
           if not p.name.endswith(".en.md") and not p.with_name(p.stem + ".en.md").exists()]
assert not missing, f".en.md karşılığı eksik: {missing}"

# 3) Türkçe alfabe sırası: c < ç < d, i(ı) < i < İ…, s < ş
words = ["Şaşkınlık", "Sözlük", "Çekirdek", "Cebir", "Dil", "Işık", "İnce", "Ilık", "Gradyan", "Ğ-test", "Ön", "Oran"]
got = sorted(words, key=lambda w: dsl._sort_key(w, "tr"))
assert got == ["Cebir", "Çekirdek", "Dil", "Gradyan", "Ğ-test", "Ilık", "Işık", "İnce", "Oran", "Ön", "Sözlük", "Şaşkınlık"], got
assert dsl._initial("ince", "tr") == "İ" and dsl._initial("ılık", "tr") == "I"

# 4) Sözlük: her terim her iki dilde ve tüm `see` bağlantıları çözülüyor
md = dsl.glossary_md(glossary, "tr")
assert "{ #ste }" in md and "(#ste)" in md
assert dsl.glossary_md(glossary, "en").count("### ") == len(glossary)

# 5) Depo süzgeci (scripts/fetch_repos.py)
api = [
    {"name": "ok", "html_url": "https://github.com/o/ok", "stargazers_count": 3, "forks_count": 1, "topics": ["nlp"]},
    {"name": "forked", "html_url": "https://github.com/o/forked", "fork": True},
    {"name": "old", "html_url": "https://github.com/o/old", "archived": True},
    {"name": "wiki", "html_url": "https://github.com/o/wiki"},
    {"name": "secret", "html_url": "https://github.com/o/secret", "private": True},
]
cfg = {"exclude": ["wiki"], "include_forks": False, "include_archived": False}
assert [r["name"] for r in fetch.select(api, cfg)] == ["ok"]
assert [r["name"] for r in fetch.select(api, {**cfg, "include_forks": True, "include_archived": True})] == ["ok", "forked", "old"]

# 6) Depo vitrini: sıralama, kaçış, güvenli bağlantı, boş durum
repos = {"payload": {"repos": [
    {"name": "a", "url": "https://github.com/o/a", "stars": 1, "pushed_at": "2026-01-01T00:00:00Z", "description": "<b>x</b>"},
    {"name": "b", "url": "https://github.com/o/b", "stars": 9, "pushed_at": "2025-01-01T00:00:00Z"},
    {"name": "c", "url": "https://github.com/o/c", "stars": 5, "pushed_at": "2026-02-01T00:00:00Z"},
    {"name": "evil", "url": "javascript:alert(1)", "stars": 99},
    {"name": "wiki", "url": "https://github.com/o/wiki", "stars": 100},
]}, "cfg": {"pinned": ["c"], "exclude": ["wiki"]}}
html = dsl.repos_html(repos, "tr", dt.date(2026, 10, 10))
order = [html.index(f">{n}</a>") for n in ("c", "b", "a")]
assert order == sorted(order), "sabitlenmiş önce, sonra yıldız sırası"
assert "<b>x</b>" not in html and "javascript:" not in html and ">wiki<" not in html
assert "dsl-lb__empty" in dsl.repos_html(None, "tr", dt.date(2026, 10, 10))
assert "dsl-lb__empty" in dsl.repos_html({"payload": {"repos": []}, "cfg": {}}, "en", dt.date(2026, 10, 10))

# 7) Okuma listesi bozuk girişi yakalıyor
try:
    dsl._https("http://x", "t")
except dsl.PluginError:
    pass
else:
    raise AssertionError("http:// bağlantısı reddedilmeliydi")
print("tüm içerik testleri geçti")
