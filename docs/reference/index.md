---
title: API Reference
subtitle: seroverse.core & seroverse.client
description: Complete API reference for the foundational SeroVerse Python package (seroverse.core and seroverse.client).
author: Tom Stanton
icon: lucide/code-xml
comments: true
tags:
  - api
  - reference
---

# API Reference

The **`seroverse`** package provides the core Structure-of-Arrays (SoA) containers, sequence/alignment/clustering engines, CSR graph & Relational MPNN solvers, and asynchronous biological repository clients shared across the SeroVerse ecosystem.

<div class="grid cards" markdown>

- :lucide-binary: **[`seroverse.core`](core.md)**

    ---

    SoA container protocols (`BatchedContainer`, `RaggedArrayContainer`), genomic `Intervals`, `GenomeAssembly`, `Alphabet` & `Sequences`, `RandstrobeIndex`, `PairwiseAligner`, `Clusterer` (MCL & CD-HIT), `CSR` & `RelationalCSR`, `Graph` & `Graphs`, `GraphAligner`, `ArborescenceSolver` (`Arborescence`, `Arborescences`), `RelationalMPNN`, and `LayoutEngine`.

-   :lucide-cloud-download: **[`seroverse.client`](client.md)**

    ---

    Token-bucket rate-limited async HTTP `Client` with `msgspec` msgpack serialization and specialized clients for **NCBI** (`NCBIClient`), **ENA** (`ENAClient`), **DSMZ BacDive & StrainInfo** (`BacDiveClient`, `StrainInfoClient`), and **Pathogenwatch** (`PathogenwatchClient`).

</div>
