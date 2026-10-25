"""Unified asynchronous biological database clients (`aiohttp` + `msgspec`)."""

from seroverse.client.base import (
    Client,
    cassette_context,
    decode_msgpack,
    encode_msgpack,
    load_msgpack,
    save_msgpack,
)
from seroverse.client.dsmz import BacDiveClient, StrainInfoClient, StrainLiterature, StrainPassport
from seroverse.client.ena import ENAAssemblyRecord, ENAClient, ENASampleRecord
from seroverse.client.ncbi import DatasetAssemblyReport, NCBIClient, NuccoreSummary, PubMedArticleId, PubMedSummary
from seroverse.client.pathogenwatch import PathogenwatchBaseClient, PathogenwatchCollection

__all__ = [
    "BacDiveClient",
    "Client",
    "DatasetAssemblyReport",
    "ENAAssemblyRecord",
    "ENAClient",
    "ENASampleRecord",
    "NCBIClient",
    "NuccoreSummary",
    "PathogenwatchBaseClient",
    "PathogenwatchCollection",
    "PubMedArticleId",
    "PubMedSummary",
    "StrainInfoClient",
    "StrainLiterature",
    "StrainPassport",
    "cassette_context",
    "decode_msgpack",
    "encode_msgpack",
    "load_msgpack",
    "save_msgpack",
]
