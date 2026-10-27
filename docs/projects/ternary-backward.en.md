---
title: Custom Backward Mechanism for Ternary Models
summary: An alternative to STE, developed because gradients break down and rounding is non-differentiable in ternary models.
status: active
order: 20
owners: []
---

# Custom Backward Mechanism for Ternary Models

!!! note "Draft page"
    This page is a skeleton built from the project short description; the maintainers will add the details.

## Summary

In [ternary](../resources/glossary.md#ternary) models, weights are represented with discrete values and the rounding step is
non-differentiable. This project develops an alternative to the [Straight-Through Estimator (STE)](../resources/glossary.md#ste)
for this problem.

## Why?

STE ignores rounding so that gradients can pass through it. However, in ternary models the
[gradients](../resources/glossary.md#gradient) can break down; that is the starting point of this project.

## Approach

_Details will be added by the project maintainers._ For background, the
[BitNet 1.58 translation](../resources/q.md) and the
[quantization section](../resources/learning.md#konu-quantization) of the reading list are useful.

## Status

Under active development.

## Roadmap

- [ ] A measurable definition of the problem
- [ ] First implementation of the alternative mechanism
- [ ] Comparison with STE

## How to join

Read the [Contributing Guide](../getting-started/contributing.md) and reach the project maintainers on Discord.
