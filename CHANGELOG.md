# Release Notes

All notable changes to **`SeroVerse`** and the SeroVerse ecosystem are documented on this page. This project adheres to [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) specification.

---

## [Unreleased]

### Added

- **Structure-of-Arrays & Graph Core (`seroverse.core`)**:
  - `BatchedContainer` and `RaggedArrayContainer` zero-copy memory protocols (`seroverse.core.collections`).
  - `Intervals` (`seroverse.core.interval`) and `GenomeAssembly` (`seroverse.core.genome`, `seroverse.core.io.fasta`).
  - 2-bit DNA/RNA (`Alphabet.DNA`) and `MMSEQS12` reduced amino-acid alphabets with ragged translation across NCBI translation tables (`seroverse.core.seq.alphabet`).
  - `RandstrobeIndex` ($O(1)$ strobemer seeding) and `FracMinHashIndex` k-mer indices (`seroverse.core.seq.kmers`).
  - Vectorized unbanded/banded Smith-Waterman-Gotoh local and semi-global alignment (`PairwiseAligner` in `seroverse.core.seq.pairwise`).
  - Hybrid sequence + motif-rescued protein clustering (`Clusterer` supporting greedy incremental and sparse Markov Clustering `MCL` modes in `seroverse.core.seq.clustering`).
  - Packed CSR directed multigraphs (`Graph`, `Graphs`), Consensus Pan-Graph compression (`GraphBatch.as_pan_graph`), Hungarian + L2-QP `GraphAligner`, branch-and-bound `maximum_spanning_arborescence`, `RelationalMPNN`, and Fruchterman-Reingold `LayoutEngine` (`seroverse.core.graph`).
- **Async Repository & Metadata Clients (`seroverse.client`)**:
  - Async token-bucket rate-limited HTTP client (`Client`) and `msgspec` cassette serialization (`cassette_context`, `encode_msgpack`, `decode_msgpack`, `load_msgpack`, `save_msgpack` in `seroverse.client.base`).
  - `NCBIClient` (`seroverse.client.ncbi`), `ENAClient` (`seroverse.client.ena`), `BacDiveClient` & `StrainInfoClient` (`seroverse.client.dsmz`), and `PathogenwatchBaseClient` (`seroverse.client.pathogenwatch`).
- **Unified Ecosystem Documentation**:
  - Unified Zensical documentation hub for `SeroVerse`, `SeroRef`, `KBank`, `SeroGen`, and `SeroEpi` with organization-wide GitHub Discussions announcements and Giscus comments.
