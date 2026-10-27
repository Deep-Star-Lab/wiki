# Contributing Guide

A short roadmap for those who want to contribute to Deep Star Lab.

## Ways to Help the Wiki

- Add a new project record
- Summarize papers
- Help with code documentation
- Mirror your own content
- Update / translate existing pages
- Report bugs and suggestions (issues)

## Steps

1. Fork the repository
2. Make your changes, install the packages and preview with `mkdocs serve`
3. Push your changes
4. Open a pull request and add a short description

## Author and translator byline

Each person is defined **once** in `data/authors.yml` (an `first-last` id and a "First Last" name). Pages use ids in the
front matter instead of typing names; the byline under the title is generated automatically:

```yaml
---
title: Article title
authors: [first-last, another-author]   # original authors
translators: [berkan-dogan]             # translators
date_translated: 2026-06-01
original_url: https://...
---
```

If an id is missing from `authors.yml`, the build fails with a clear error. Use the [Glossary](../resources/glossary.md) for terminology in translations.
To get your contribution onto the [Leaderboard](../leaderboard.md), add an entry with proof to `data/contributions.yml`.
See the [Rules](../rules.md) page for the general rules.

!!! note
    Turkish is the default language. Don't forget to add the English
    counterpart of each page with the `.en.md` extension.
