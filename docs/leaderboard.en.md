# Leaderboard

Contributions by Deep Star Lab researchers. Ranking is by **points**; points are computed
automatically from the ledger entries using the rules below.

<!-- dsl:leaderboard top=50 -->

## How points are earned

<!-- dsl:scoring -->

- When a contribution has several people, the points are **split equally**.
- For types with a "monthly cap", one person cannot earn more than the cap from that type per month (this prevents spam).
- The reviewer may apply a quality `weight` multiplier (0.5–2.0).
- The **This month** and **Last 90 days** tabs keep newcomers visible; the all-time table preserves long-term effort.

## Recent contributions

<!-- dsl:ledger n=15 -->

## How to add your contribution

1. If you are not in `data/authors.yml`, add yourself with an `first-last` id.
2. Add a new entry to `data/contributions.yml` (type, title, date and a **proof link** are required).
3. Open a pull request. Once a reviewer verifies the proof and merges it, your contribution shows up here.

```yaml
- people: [first-last]
  type: translation        # paper | release | project | translation | code | summary | docs | review | issue
  title: "Short description"
  date: 2026-10-10
  url: https://github.com/Deep-Star-Lab/wiki/pull/123
```

The ledger is read on every build; a wrong id, type or date stops the build with a clear error message.
