"""Deep Star Lab — MkDocs hook.

İki iş yapar, ikisi de `data/` altındaki aynı kayıtlardan beslenir:

1. Yazar standardı
   Sayfa front matter'ında `authors:` / `translators:` kişi id'leri yazılır
   (bkz. data/authors.yml). Hook, başlığın hemen altına standart bir künye
   ekler. Tanımsız id derleme hatasıdır.

2. Liderlik tablosu
   data/contributions.yml (katkı defteri) + data/scoring.yml (puan kuralları)
   okunur; sayfalarda şu yer tutucular tabloya dönüşür:

       <!-- dsl:leaderboard top=5 -->   Tüm zamanlar / Bu ay / Son 90 gün sekmeleri
       <!-- dsl:scoring -->             puan kuralları tablosu
       <!-- dsl:ledger n=15 -->         son katkılar listesi
       <!-- dsl:glossary -->            sözlük (data/glossary.yml)
       <!-- dsl:reading -->             okuma listesi (data/reading_list.yml)
       <!-- dsl:repos -->               GitHub depo vitrini (data/repos.json, scripts/fetch_repos.py)
       <!-- dsl:projects -->            docs/projects/*.md front matter'ından proje kartları

Proje sayfalarında `status`, `owners`, `repo`, `hf` front matter alanları künyeye eklenir.

Dil, aktif Material dilinden alınır (i18n eklentisi her derlemede ayarlar).
Test için bugünün tarihi `DSL_TODAY=YYYY-MM-DD` ile sabitlenebilir.
"""

from __future__ import annotations

import datetime as dt
import html
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError

log = logging.getLogger("mkdocs.hooks.dsl")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ID_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

TEXT = {
    "tr": {
        "authors": "Yazarlar",
        "translators": "Çevirmenler",
        "translated": "Çeviri tarihi",
        "original": "Orijinal makale",
        "all": "Tüm zamanlar",
        "month": "Bu ay",
        "d90": "Son 90 gün",
        "person": "Araştırmacı",
        "points": "Puan",
        "contrib": "Katkılar",
        "empty": "Bu dönemde henüz katkı yok.",
        "type": "Tür",
        "worth": "Puan",
        "cap": "Aylık üst sınır",
        "when": "Ne zaman?",
        "none": "—",
        "date": "Tarih",
        "what": "Katkı",
        "who": "Kişi",
        "owners": "Sorumlular",
        "status": "Durum",
        "links": "Bağlantılar",
        "repo": "Depo",
        "hf": "Hugging Face",
        "st": {"active": "Aktif", "planned": "Planlanıyor", "paused": "Beklemede", "done": "Tamamlandı"},
        "level": {"beginner": "Başlangıç", "intermediate": "Orta", "advanced": "İleri"},
        "see": "Ayrıca bakınız",
        "updated": "Güncellendi",
        "no_repos": "Depo listesi henüz oluşturulmadı. Site derlenirken GitHub organizasyonundan çekilir.",
        "no_projects": "Henüz proje sayfası yok.",
        "paper": "Makale",
        "level_h": "Seviye",
        "why_h": "Neden okunmalı?",
        "months": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
                   "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
    },
    "en": {
        "authors": "Authors",
        "translators": "Translators",
        "translated": "Translated on",
        "original": "Original article",
        "all": "All time",
        "month": "This month",
        "d90": "Last 90 days",
        "person": "Researcher",
        "points": "Points",
        "contrib": "Contributions",
        "empty": "No contributions in this period yet.",
        "type": "Type",
        "worth": "Points",
        "cap": "Monthly cap",
        "when": "When to use",
        "none": "—",
        "date": "Date",
        "what": "Contribution",
        "who": "People",
        "owners": "Maintainers",
        "status": "Status",
        "links": "Links",
        "repo": "Repository",
        "hf": "Hugging Face",
        "st": {"active": "Active", "planned": "Planned", "paused": "Paused", "done": "Done"},
        "level": {"beginner": "Beginner", "intermediate": "Intermediate", "advanced": "Advanced"},
        "see": "See also",
        "updated": "Updated",
        "no_repos": "The repository list has not been generated yet. It is fetched from the GitHub organization during the site build.",
        "no_projects": "No project pages yet.",
        "paper": "Paper",
        "level_h": "Level",
        "why_h": "Why read it?",
        "months": ["January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December"],
    },
}

_state: dict = {}


# --------------------------------------------------------------------------- #
# Veri yükleme ve doğrulama
# --------------------------------------------------------------------------- #
def _load(name: str):
    path = DATA / name
    try:
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise PluginError(f"[dsl] Veri dosyası yok: {path}") from exc
    except yaml.YAMLError as exc:
        raise PluginError(f"[dsl] {name} okunamadı: {exc}") from exc


def _as_date(value, where: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise PluginError(f"[dsl] {where}: geçersiz tarih {value!r} (YYYY-MM-DD bekleniyor)") from exc


def _today() -> dt.date:
    override = os.environ.get("DSL_TODAY")
    return dt.date.fromisoformat(override) if override else dt.date.today()


def load_data() -> dict:
    people = _load("authors.yml")
    scoring = _load("scoring.yml").get("types") or {}
    ledger = _load("contributions.yml") or []

    for pid, p in people.items():
        if not ID_RE.match(str(pid)):
            raise PluginError(f"[dsl] authors.yml: geçersiz id {pid!r} (ad-soyad, küçük harf, ASCII)")
        if not isinstance(p, dict) or not str(p.get("name", "")).strip():
            raise PluginError(f"[dsl] authors.yml: {pid!r} için `name` zorunlu")

    for tid, t in scoring.items():
        if not isinstance(t.get("points"), (int, float)) or t["points"] < 0:
            raise PluginError(f"[dsl] scoring.yml: {tid!r} için `points` sayı olmalı")
        for lang in TEXT:
            if lang not in (t.get("label") or {}):
                raise PluginError(f"[dsl] scoring.yml: {tid!r} için label.{lang} eksik")

    entries = []
    if not isinstance(ledger, list):
        raise PluginError("[dsl] contributions.yml bir liste olmalı")
    for i, e in enumerate(ledger, 1):
        where = f"contributions.yml #{i}"
        for key in ("people", "type", "title", "date", "url"):
            if not e.get(key):
                raise PluginError(f"[dsl] {where}: `{key}` zorunlu")
        if e["type"] not in scoring:
            raise PluginError(f"[dsl] {where}: bilinmeyen tür {e['type']!r} (olanlar: {', '.join(scoring)})")
        ids = e["people"] if isinstance(e["people"], list) else [e["people"]]
        for pid in ids:
            if pid not in people:
                raise PluginError(f"[dsl] {where}: {pid!r} data/authors.yml içinde tanımlı değil")
        weight = float(e.get("weight", 1.0))
        if not 0.5 <= weight <= 2.0:
            raise PluginError(f"[dsl] {where}: weight 0.5–2.0 aralığında olmalı")
        entries.append({
            "people": ids, "type": e["type"], "title": str(e["title"]),
            "date": _as_date(e["date"], where), "url": str(e["url"]), "weight": weight,
        })
    entries.sort(key=lambda x: x["date"])
    return {"people": people, "scoring": scoring, "entries": entries}


# --------------------------------------------------------------------------- #
# Puan hesabı
# --------------------------------------------------------------------------- #
def award(data: dict, today: dt.date):
    """Her (kişi, katkı) için verilen puanı döndürür; aylık üst sınırlar uygulanır."""
    used: dict = defaultdict(float)  # (kişi, tür, yıl-ay) -> verilen puan
    rows = []
    for e in data["entries"]:
        if e["date"] > today:
            continue
        spec = data["scoring"][e["type"]]
        share = spec["points"] * e["weight"] / len(e["people"])
        cap = spec.get("monthly_cap")
        for pid in e["people"]:
            pts = share
            if cap is not None:
                key = (pid, e["type"], e["date"].strftime("%Y-%m"))
                pts = max(0.0, min(share, cap - used[key]))
                used[key] += pts
            rows.append({"person": pid, "type": e["type"], "date": e["date"],
                         "title": e["title"], "points": pts})
    return rows


def standings(data: dict, today: dt.date, period: str):
    rows = award(data, today)
    if period == "month":
        rows = [r for r in rows if (r["date"].year, r["date"].month) == (today.year, today.month)]
    elif period == "d90":
        rows = [r for r in rows if today - dt.timedelta(days=90) <= r["date"] <= today]
    by_person: dict = {}
    for r in rows:
        s = by_person.setdefault(r["person"], {"person": r["person"], "points": 0.0,
                                               "count": 0, "types": defaultdict(int), "last": r["date"]})
        s["points"] += r["points"]
        s["count"] += 1
        s["types"][r["type"]] += 1
        s["last"] = max(s["last"], r["date"])
    ranked = sorted(by_person.values(),
                    key=lambda s: (-s["points"], -s["count"], data["people"][s["person"]]["name"]))
    # eşit puan = eşit sıra (1, 2, 2, 4)
    prev, rank = None, 0
    for i, s in enumerate(ranked, 1):
        if s["points"] != prev:
            rank, prev = i, s["points"]
        s["rank"] = rank
    return ranked


# --------------------------------------------------------------------------- #
# HTML üretimi
# --------------------------------------------------------------------------- #
def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _fmt_points(p: float) -> str:
    return f"{p:g}" if p == int(p) else f"{p:.1f}"


def _fmt_date(d: dt.date, lang: str) -> str:
    t = TEXT[lang]
    return f"{d.day} {t['months'][d.month - 1]} {d.year}"


def _profile_url(p: dict) -> str | None:
    if p.get("url"):
        return p["url"]
    if p.get("github"):
        return f"https://github.com/{p['github']}"
    if p.get("hf"):
        return f"https://huggingface.co/{p['hf']}"
    return None


def person_link(p: dict) -> str:
    url = _profile_url(p)
    name = _e(p["name"])
    return f'<a href="{_e(url)}">{name}</a>' if url else name


def _avatar(p: dict) -> str:
    if p.get("github"):
        return (f'<img class="dsl-lb__avatar" src="https://github.com/{_e(p["github"])}.png?size=64" '
                f'alt="" width="28" height="28" loading="lazy">')
    initial = _e(p["name"].strip()[:1].upper())
    return f'<span class="dsl-lb__avatar dsl-lb__avatar--initial" aria-hidden="true">{initial}</span>'


def leaderboard_table(data: dict, today: dt.date, period: str, top: int, lang: str) -> str:
    t = TEXT[lang]
    ranked = standings(data, today, period)[:top]
    if not ranked:
        return f'<p class="dsl-lb__empty">{_e(t["empty"])}</p>'
    out = ['<div class="dsl-lb"><table>',
           f'<thead><tr><th>#</th><th>{_e(t["person"])}</th><th>{_e(t["contrib"])}</th>'
           f'<th class="dsl-lb__num">{_e(t["points"])}</th></tr></thead><tbody>']
    for s in ranked:
        p = data["people"][s["person"]]
        chips = "".join(
            f'<span class="dsl-lb__chip">{_e(data["scoring"][k]["label"][lang])} ×{n}</span>'
            for k, n in sorted(s["types"].items(), key=lambda kv: -kv[1]))
        out.append(
            f'<tr class="dsl-lb__row dsl-lb__row--r{min(s["rank"], 4)}">'
            f'<td class="dsl-lb__rank">{s["rank"]}</td>'
            f'<td class="dsl-lb__who">{_avatar(p)}<span>{person_link(p)}</span></td>'
            f'<td>{chips}</td>'
            f'<td class="dsl-lb__num"><strong>{_fmt_points(s["points"])}</strong></td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def leaderboard_md(data: dict, today: dt.date, top: int, lang: str) -> str:
    """Material sekmeli içerik (pymdownx.tabbed) olarak üç dönem."""
    t = TEXT[lang]
    blocks = []
    for period in ("all", "month", "d90"):
        table = leaderboard_table(data, today, period, top, lang)
        blocks.append(f'=== "{t[period]}"\n\n    {table}\n')
    return "\n".join(blocks)


def scoring_table(data: dict, lang: str) -> str:
    t = TEXT[lang]
    out = ['<div class="dsl-lb"><table>',
           f'<thead><tr><th>{_e(t["type"])}</th><th class="dsl-lb__num">{_e(t["worth"])}</th>'
           f'<th class="dsl-lb__num">{_e(t["cap"])}</th><th>{_e(t["when"])}</th></tr></thead><tbody>']
    for tid, spec in sorted(data["scoring"].items(), key=lambda kv: -kv[1]["points"]):
        cap = spec.get("monthly_cap")
        out.append(
            f'<tr><td>{_e(spec["label"][lang])}</td>'
            f'<td class="dsl-lb__num">{_fmt_points(spec["points"])}</td>'
            f'<td class="dsl-lb__num">{_fmt_points(cap) if cap is not None else _e(t["none"])}</td>'
            f'<td>{_e((spec.get("about") or {}).get(lang, ""))}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def ledger_list(data: dict, today: dt.date, n: int, lang: str) -> str:
    t = TEXT[lang]
    entries = [e for e in data["entries"] if e["date"] <= today][-n:][::-1]
    if not entries:
        return f'<p class="dsl-lb__empty">{_e(t["empty"])}</p>'
    out = ['<div class="dsl-lb"><table>',
           f'<thead><tr><th>{_e(t["date"])}</th><th>{_e(t["what"])}</th><th>{_e(t["who"])}</th></tr></thead><tbody>']
    for e in entries:
        names = ", ".join(person_link(data["people"][pid]) for pid in e["people"])
        label = data["scoring"][e["type"]]["label"][lang]
        out.append(
            f'<tr><td>{_e(_fmt_date(e["date"], lang))}</td>'
            f'<td><span class="dsl-lb__chip">{_e(label)}</span> '
            f'<a href="{_e(e["url"])}">{_e(e["title"])}</a></td><td>{names}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def byline(meta: dict, people: dict, lang: str, page_src: str) -> str | None:
    t = TEXT[lang]
    lines = []
    status = meta.get("status")
    if status:
        if status not in STATUSES:
            raise PluginError(f"[dsl] {page_src}: status {', '.join(STATUSES)} değerlerinden biri olmalı")
        lines.append(f'<p><strong>{_e(t["status"])}:</strong> '
                     f'<span class="dsl-status dsl-status--{status}">{_e(t["st"][status])}</span></p>')
    for key in ("owners", "authors", "translators"):
        ids = meta.get(key)
        if not ids:
            continue
        ids = ids if isinstance(ids, list) else [ids]
        names = []
        for pid in ids:
            if pid not in people:
                raise PluginError(
                    f"[dsl] {page_src}: front matter `{key}` içindeki {pid!r} data/authors.yml'de yok. "
                    "Kişiyi önce oraya ekleyin.")
            names.append(person_link(people[pid]))
        lines.append(f'<p><strong>{_e(t[key])}:</strong> {", ".join(names)}</p>')
    link_bits = []
    for key in ("repo", "hf"):
        if meta.get(key):
            link_bits.append(f'<a href="{_e(_https(meta[key], page_src + " " + key))}">{_e(t[key])}</a>')
    if link_bits:
        lines.append(f'<p><strong>{_e(t["links"])}:</strong> {" · ".join(link_bits)}</p>')
    meta_bits = []
    if meta.get("date_translated"):
        d = _as_date(meta["date_translated"], f"{page_src} date_translated")
        meta_bits.append(f'{_e(t["translated"])}: {_e(_fmt_date(d, lang))}')
    if meta.get("original_url"):
        meta_bits.append(f'<a href="{_e(meta["original_url"])}">{_e(t["original"])}</a>')
    if meta_bits:
        lines.append(f'<p class="dsl-byline__meta">{" · ".join(meta_bits)}</p>')
    if not lines:
        return None
    return '<div class="dsl-byline">' + "".join(lines) + "</div>"



# --------------------------------------------------------------------------- #
# Sözlük, okuma listesi, depo vitrini, proje kartları
# --------------------------------------------------------------------------- #
TR_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
TR_ORDER = {c: i for i, c in enumerate(TR_ALPHABET)}
LEVELS = ("beginner", "intermediate", "advanced")
STATUSES = ("active", "planned", "paused", "done")
LANG_COLORS = {"Python": "#3572A5", "Jupyter Notebook": "#DA5B0B", "C++": "#f34b7d", "C": "#555555",
               "Rust": "#dea584", "JavaScript": "#f1e05a", "TypeScript": "#3178c6", "Go": "#00ADD8",
               "Cuda": "#3A4E3A", "Shell": "#89e051", "HTML": "#e34c26", "Julia": "#a270ba"}


def _fold(s: str, lang: str) -> str:
    if lang == "tr":
        s = s.replace("I", "ı").replace("İ", "i")
    return s.lower()


def _sort_key(s: str, lang: str):
    s = _fold(s, lang)
    if lang == "tr":
        return [TR_ORDER.get(c, 100 + ord(c)) for c in s]
    return list(s)


def _initial(s: str, lang: str) -> str:
    c = s.strip()[:1]
    if lang == "tr":
        return {"i": "İ", "ı": "I"}.get(c, c.upper())
    return c.upper()


def _letter_id(letter: str) -> str:
    return "harf-" + (letter.lower() if letter.isascii() and letter.isalnum() else f"u{ord(letter):x}")


def _https(url: str, where: str) -> str:
    if not str(url).startswith("https://"):
        raise PluginError(f"[dsl] {where}: bağlantı https:// ile başlamalı: {url!r}")
    return str(url)


def load_glossary() -> list:
    entries = _load("glossary.yml")
    if not isinstance(entries, list):
        raise PluginError("[dsl] glossary.yml bir liste olmalı")
    ids = set()
    for i, e in enumerate(entries, 1):
        where = f"glossary.yml #{i}"
        if not ID_RE.match(str(e.get("id", ""))):
            raise PluginError(f"[dsl] {where}: geçersiz id {e.get('id')!r}")
        if e["id"] in ids:
            raise PluginError(f"[dsl] {where}: yinelenen id {e['id']!r}")
        ids.add(e["id"])
        for k in ("tr", "en"):
            if not str(e.get(k, "")).strip():
                raise PluginError(f"[dsl] {where} ({e['id']}): `{k}` zorunlu")
            if not str((e.get("def") or {}).get(k, "")).strip():
                raise PluginError(f"[dsl] {where} ({e['id']}): def.{k} zorunlu")
    for e in entries:
        for ref in e.get("see") or []:
            if ref not in ids:
                raise PluginError(f"[dsl] glossary.yml ({e['id']}): see içindeki {ref!r} tanımlı değil")
    return entries


def load_reading() -> dict:
    data = _load("reading_list.yml")
    topics, items = data.get("topics") or {}, data.get("items") or []
    for tid, t in topics.items():
        for lang in TEXT:
            if lang not in t:
                raise PluginError(f"[dsl] reading_list.yml: topics.{tid}.{lang} eksik")
    for i, it in enumerate(items, 1):
        where = f"reading_list.yml #{i}"
        for key in ("title", "authors", "year", "url", "topic", "level", "why"):
            if not it.get(key):
                raise PluginError(f"[dsl] {where}: `{key}` zorunlu")
        _https(it["url"], where)
        if it["topic"] not in topics:
            raise PluginError(f"[dsl] {where}: bilinmeyen topic {it['topic']!r}")
        if it["level"] not in LEVELS:
            raise PluginError(f"[dsl] {where}: level {', '.join(LEVELS)} değerlerinden biri olmalı")
        for lang in TEXT:
            if lang not in it["why"]:
                raise PluginError(f"[dsl] {where}: why.{lang} eksik")
    return {"topics": topics, "items": items}


def load_repos():
    """scripts/fetch_repos.py çıktısı; yoksa None (sayfa boş durum gösterir, derleme bozulmaz)."""
    import json
    path = DATA / "repos.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PluginError(f"[dsl] repos.json okunamadı: {exc}") from exc
    cfg = _load("repos.yml")
    return {"payload": payload, "cfg": cfg}


def glossary_md(entries: list, lang: str) -> str:
    other = "en" if lang == "tr" else "tr"
    by_id = {e["id"]: e for e in entries}
    ordered = sorted(entries, key=lambda e: _sort_key(e[lang], lang))
    letters = []
    for e in ordered:
        L = _initial(e[lang], lang)
        if L not in letters:
            letters.append(L)
    out = [" · ".join(f"[{L}](#{_letter_id(L)})" for L in letters), ""]
    current = None
    for e in ordered:
        L = _initial(e[lang], lang)
        if L != current:
            current = L
            out += [f"## {L} {{ #{_letter_id(L)} }}", ""]
        out += [f"### {e[lang]} {{ #{e['id']} }}", "", f"*{e[other]}*", "", e["def"][lang].strip(), ""]
        refs = [f"[{by_id[r][lang]}](#{r})" for r in e.get("see") or []]
        if refs:
            out += [f"**{TEXT[lang]['see']}:** " + ", ".join(refs), ""]
    return "\n".join(out)


def reading_md(data: dict, people: dict, lang: str) -> str:
    t = TEXT[lang]
    out = []
    for tid, topic in data["topics"].items():
        rows = [it for it in data["items"] if it["topic"] == tid]
        if not rows:
            continue
        rows.sort(key=lambda it: (LEVELS.index(it["level"]), it["year"]))
        out += [f"### {topic[lang]} {{ #konu-{tid} }}", ""]
        html_rows = []
        for it in rows:
            who = ""
            if it.get("suggested_by"):
                for pid in it["suggested_by"]:
                    if pid not in people:
                        raise PluginError(f"[dsl] reading_list.yml: suggested_by içindeki {pid!r} authors.yml'de yok")
                who = " · " + ", ".join(person_link(people[pid]) for pid in it["suggested_by"])
            html_rows.append(
                f'<tr><td><a href="{_e(it["url"])}">{_e(it["title"])}</a><br>'
                f'<small>{_e(it["authors"])}, {_e(it["year"])}{who}</small></td>'
                f'<td><span class="dsl-lb__chip dsl-level dsl-level--{it["level"]}">{_e(t["level"][it["level"]])}</span></td>'
                f'<td>{_e(it["why"][lang])}</td></tr>')
        out += ['<div class="dsl-lb"><table><thead><tr>'
                f'<th>{_e(t["paper"])}</th><th>{_e(t["level_h"])}</th><th>{_e(t["why_h"])}</th>'
                '</tr></thead><tbody>' + "".join(html_rows) + "</tbody></table></div>", ""]
    return "\n".join(out)


def repos_html(repos_data, lang: str, today: dt.date) -> str:
    t = TEXT[lang]
    if not repos_data or not repos_data["payload"].get("repos"):
        return f'<p class="dsl-lb__empty">{_e(t["no_repos"])}</p>'
    cfg, repos = repos_data["cfg"], list(repos_data["payload"]["repos"])
    pinned = [str(x) for x in (cfg.get("pinned") or [])]
    excluded = set(cfg.get("exclude") or [])
    repos = [r for r in repos if r["name"] not in excluded]

    # Kararlı sıralamalar, en önemsiz anahtardan en önemliye: ad → son güncelleme → yıldız → sabitlenmiş
    pin_rank = lambda r: pinned.index(r["name"]) if r["name"] in pinned else len(pinned)
    repos.sort(key=lambda r: r["name"])
    repos.sort(key=lambda r: str(r.get("pushed_at") or ""), reverse=True)
    repos.sort(key=lambda r: int(r.get("stars", 0)), reverse=True)
    repos.sort(key=pin_rank)

    cards = []
    for r in repos:
        url = str(r.get("url", ""))
        if not url.startswith("https://github.com/"):
            continue
        lang_name = r.get("language")
        dot = (f'<span class="dsl-repo__lang"><i style="background:{LANG_COLORS.get(lang_name, "#8b949e")}"></i>'
               f'{_e(lang_name)}</span>') if lang_name else ""
        topics = "".join(f'<span class="dsl-lb__chip">{_e(x)}</span>' for x in (r.get("topics") or [])[:5])
        updated = ""
        if r.get("pushed_at"):
            try:
                d = dt.date.fromisoformat(str(r["pushed_at"])[:10])
                updated = f'<span>{_e(t["updated"])}: {_e(_fmt_date(d, lang))}</span>'
            except ValueError:
                pass
        lic = f'<span>{_e(r["license"])}</span>' if r.get("license") and r["license"] != "NOASSERTION" else ""
        cards.append(
            '<article class="dsl-repo">'
            f'<h3><a href="{_e(url)}">{_e(r["name"])}</a></h3>'
            f'<p>{_e(r.get("description") or "")}</p>'
            f'<div class="dsl-repo__topics">{topics}</div>'
            f'<div class="dsl-repo__meta">{dot}<span>★ {int(r.get("stars", 0))}</span>'
            f'<span>⑂ {int(r.get("forks", 0))}</span>{lic}{updated}</div></article>')
    if not cards:
        return f'<p class="dsl-lb__empty">{_e(t["no_repos"])}</p>'
    return '<div class="dsl-repos">' + "".join(cards) + "</div>"


_FM = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


def _front_matter(path: Path) -> dict:
    m = _FM.match(path.read_text(encoding="utf-8"))
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise PluginError(f"[dsl] {path.name}: front matter okunamadı: {exc}") from exc


def projects_md(docs_dir: str, lang: str) -> str:
    t = TEXT[lang]
    pdir = Path(docs_dir) / "projects"
    cards = []
    for f in sorted(pdir.glob("*.md")):
        if f.name.startswith("index") or re.search(r"\.[a-z]{2}\.md$", f.name):
            continue
        slug = f.stem
        src = pdir / f"{slug}.{lang}.md"
        meta = _front_matter(src if lang != "tr" and src.exists() else f)
        for key in ("title", "summary", "status"):
            if not meta.get(key):
                raise PluginError(f"[dsl] projects/{f.name}: front matter `{key}` zorunlu")
        if meta["status"] not in STATUSES:
            raise PluginError(f"[dsl] projects/{f.name}: status {', '.join(STATUSES)} değerlerinden biri olmalı")
        cards.append((int(meta.get("order", 100)), slug, meta))
    if not cards:
        return f'<p class="dsl-lb__empty">{_e(t["no_projects"])}</p>'
    out = ['<div class="grid cards" markdown>', ""]
    for _, slug, meta in sorted(cards, key=lambda c: (c[0], c[1])):
        out += [f"-   **[{meta['title']}]({slug}.md)**", "", "    ---", "",
                f"    {str(meta['summary']).strip()}", "",
                f'    <span class="dsl-status dsl-status--{meta["status"]}">{_e(t["st"][meta["status"]])}</span>', ""]
    out.append("</div>")
    return "\n".join(out)

# --------------------------------------------------------------------------- #
# MkDocs kancaları
# --------------------------------------------------------------------------- #
def on_config(config):
    _state["data"] = load_data()
    _state["glossary"] = load_glossary()
    _state["reading"] = load_reading()
    _state["repos"] = load_repos()
    return config


def on_serve(server, config, builder):
    server.watch(str(DATA))  # veri değişince canlı önizleme yenilensin
    return server


_TOKEN = re.compile(r"<!--\s*dsl:(leaderboard|scoring|ledger|glossary|reading|repos|projects)((?:\s+\w+=\d+)*)\s*-->")
_H1 = re.compile(r"^# .+$", re.M)


def _lang(config) -> str:
    lang = str(config.theme.get("language") or "tr").split("-")[0]
    return lang if lang in TEXT else "tr"


def on_page_markdown(markdown, page, config, files):
    data = _state.get("data")
    if data is None:  # on_config çalışmadıysa (ör. tek başına test)
        on_config(config)
        data = _state["data"]
    lang, today = _lang(config), _today()

    def render(m):
        kind = m.group(1)
        args = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(2) or "")}
        if kind == "leaderboard":
            return leaderboard_md(data, today, args.get("top", 10), lang)
        if kind == "scoring":
            return scoring_table(data, lang)
        if kind == "glossary":
            return glossary_md(_state["glossary"], lang)
        if kind == "reading":
            return reading_md(_state["reading"], data["people"], lang)
        if kind == "repos":
            return repos_html(_state["repos"], lang, today)
        if kind == "projects":
            return projects_md(config.docs_dir, lang)
        return ledger_list(data, today, args.get("n", 15), lang)

    markdown = _TOKEN.sub(render, markdown)

    block = byline(page.meta, data["people"], lang, page.file.src_path)
    if block:
        h1 = _H1.search(markdown)
        if h1:
            markdown = markdown[:h1.end()] + "\n\n" + block + "\n" + markdown[h1.end():]
        else:
            markdown = block + "\n\n" + markdown
    return markdown
