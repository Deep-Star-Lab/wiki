"""hooks/dsl.py için hızlı testler.  Çalıştır:  python3 tests/test_dsl.py"""
import datetime as dt
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("dsl", Path(__file__).resolve().parent.parent / "hooks" / "dsl.py")
dsl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsl)

D = dt.date
PEOPLE = {k: {"name": k.title()} for k in ("a", "b", "c")}
SCORING = {
    "paper": {"points": 100, "label": {"tr": "M", "en": "P"}},
    "code": {"points": 15, "monthly_cap": 30, "label": {"tr": "K", "en": "C"}},
}


def mk(*entries):
    return {"people": PEOPLE, "scoring": SCORING, "entries": sorted(entries, key=lambda e: e["date"])}


def e(people, typ, date, weight=1.0):
    return {"people": people, "type": typ, "title": "t", "date": date, "url": "u", "weight": weight}


def pts(data, today, period="all"):
    return {s["person"]: s["points"] for s in dsl.standings(data, today, period)}


today = D(2026, 10, 10)

# eşit bölüşüm + ağırlık çarpanı
assert pts(mk(e(["a", "b"], "paper", D(2026, 1, 1))), today) == {"a": 50, "b": 50}
assert pts(mk(e(["a"], "paper", D(2026, 1, 1), 1.5)), today) == {"a": 150}

# aylık üst sınır: 3 PR = 45 ama sınır 30; ertesi ay yeniden başlar
d = mk(e(["a"], "code", D(2026, 9, 1)), e(["a"], "code", D(2026, 9, 2)), e(["a"], "code", D(2026, 9, 3)),
       e(["a"], "code", D(2026, 10, 1)))
assert pts(d, today) == {"a": 45}, pts(d, today)          # 30 (eylül) + 15 (ekim)
assert pts(d, today, "month") == {"a": 15}

# gelecek tarihli giriş sayılmaz
assert pts(mk(e(["a"], "paper", D(2027, 1, 1))), today) == {}

# 90 gün penceresi sınırı (dahil)
d = mk(e(["a"], "paper", today - dt.timedelta(days=90)), e(["b"], "paper", today - dt.timedelta(days=91)))
assert pts(d, today, "d90") == {"a": 100}

# eşitlikte aynı sıra, sonraki sıra atlanır (1, 1, 3)
d = mk(e(["a"], "paper", D(2026, 1, 1)), e(["b"], "paper", D(2026, 1, 2)), e(["c"], "code", D(2026, 1, 3)))
assert [s["rank"] for s in dsl.standings(d, today, "all")] == [1, 1, 3]

# XSS: isimler kaçırılır
evil = {"x": {"name": '<script>alert(1)</script>', "github": 'a"b'}}
assert "<script>" not in dsl.person_link(evil["x"]) and "<script>" not in dsl._avatar(evil["x"])

# boş dönem
assert "dsl-lb__empty" in dsl.leaderboard_table(mk(), today, "month", 5, "tr")
print("tüm testler geçti")
