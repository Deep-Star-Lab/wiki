#!/usr/bin/env python3
"""GitHub organizasyonunun herkese açık depolarını data/repos.json dosyasına yazar.

Kullanım:   python scripts/fetch_repos.py [--strict]
Ortam:      GITHUB_TOKEN / GH_TOKEN (opsiyonel; verilirse rate limit yükselir)

Ayarlar data/repos.yml içindedir. Ağ/API hatasında (--strict yoksa) uyarı verir,
mevcut data/repos.json'a dokunmaz ve 0 ile çıkar; böylece site derlemesi bozulmaz.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "data" / "repos.yml"
OUT = ROOT / "data" / "repos.json"
API = "https://api.github.com/orgs/{org}/repos?per_page=100&type=public&page={page}"


def fetch_page(org: str, page: int, token: str | None) -> list[dict]:
    req = urllib.request.Request(API.format(org=org, page=page), headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "deep-star-lab-wiki",
        "X-GitHub-Api-Version": "2022-11-28",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def select(items: list[dict], cfg: dict) -> list[dict]:
    """API yanıtını ayarlara göre süzer ve yalnızca gereken alanları tutar."""
    out = []
    for r in items:
        if r.get("private") or r.get("name") in set(cfg.get("exclude") or []):
            continue
        if r.get("fork") and not cfg.get("include_forks"):
            continue
        if r.get("archived") and not cfg.get("include_archived"):
            continue
        out.append({
            "name": r["name"],
            "url": r["html_url"],
            "description": r.get("description") or "",
            "language": r.get("language"),
            "stars": int(r.get("stargazers_count") or 0),
            "forks": int(r.get("forks_count") or 0),
            "topics": list(r.get("topics") or []),
            "license": (r.get("license") or {}).get("spdx_id"),
            "homepage": r.get("homepage") or "",
            "pushed_at": r.get("pushed_at"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="hata olursa sıfırdan farklı kodla çık")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    org = cfg.get("org")
    if not org:
        print("data/repos.yml: `org` zorunlu", file=sys.stderr)
        return 1
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    items: list[dict] = []
    try:
        for page in range(1, 11):  # en fazla 1000 depo
            chunk = fetch_page(org, page, token)
            items += chunk
            if len(chunk) < 100:
                break
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"UYARI: {org} depoları alınamadı ({exc}); mevcut dosya korundu.", file=sys.stderr)
        return 1 if args.strict else 0

    repos = select(items, cfg)
    payload = {"org": org,
               "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "repos": repos}
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    print(f"{len(repos)} depo yazıldı → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
