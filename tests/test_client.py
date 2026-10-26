"""Unit tests for `seroverse.client` models, `msgspec` serialization, and async clients."""

from pathlib import Path

import msgspec

from seroverse.client.base import Client
from seroverse.client.dsmz import BacDiveClient, StrainInfoClient, StrainPassport
from seroverse.client.ena import ENAAssemblyRecord, ENAClient, ENASampleRecord
from seroverse.client.ncbi import NCBIClient, NuccoreSummary, PubMedSummary
from seroverse.client.pathogenwatch import PathogenwatchClient, PathogenwatchCollection


def test_msgpack_roundtrip_and_frozen_dataclasses(tmp_path: Path) -> None:
    ref = ENASampleRecord(
        sample_accession="SAMEA3231288",
        scientific_name="Escherichia coli",
        strain="NCTC 9001",
        serotype="O1:K1(L1):H7",
        tax_id="562",
    )
    packed = Client.encode_msgpack([ref])
    unpacked = Client.decode_msgpack(packed, list[ENASampleRecord])
    assert unpacked == [ref]

    cache_file = tmp_path / "nctc.msgpack"
    Client.save_msgpack(cache_file, [ref])
    loaded = Client.load_msgpack(cache_file, list[ENASampleRecord])
    assert loaded == [ref]


def test_strain_passport_from_raw_dict() -> None:
    raw = {
        "strain": {
            "id": 108168,
            "bacdive": 143413,
            "type_strain": False,
            "taxon": {"name": "Proteus mirabilis", "ncbi": 584},
            "relation": {
                "culture": [{"strain_number": "CCUG 19011"}, {"strain_number": "PrK 38/57"}],
                "designation": ["Prague 38/57"],
            },
            "literature": [
                {"id": 1, "pubmed": "15247885", "doi": "10.1016/j.carres.2004.05.001", "year": 2004},
            ],
            "sequences": [
                {"accession": "KY710701", "type": "sequence"},
                {"accession": "GCA_900455005.1", "type": "genome"},
            ],
        }
    }
    passport = StrainPassport.from_raw_dict(raw)
    assert passport is not None
    assert passport.si_id == 108168
    assert passport.taxon_name == "Proteus mirabilis"
    assert passport.ncbi_taxid == 584
    assert passport.bacdive_id == 143413
    assert passport.designations == ("CCUG 19011", "PrK 38/57", "Prague 38/57")
    assert passport.pmids == (15247885,)
    assert passport.dois == ("10.1016/j.carres.2004.05.001",)
    assert passport.sequence_accessions == ("KY710701",)
    assert passport.genome_accessions == ("GCA_900455005.1",)


def test_ena_and_ncbi_json_decoding() -> None:
    sample_json = b'[{"sample_accession":"SAMEA1","scientific_name":"Escherichia coli","strain":"NCTC9001","serotype":"O1:K1:H7","serovar":"","tax_id":"562"}]'
    samples = msgspec.json.decode(sample_json, type=list[ENASampleRecord])
    assert len(samples) == 1
    assert samples[0].sample_accession == "SAMEA1"
    assert samples[0].serotype == "O1:K1:H7"

    asm_json = b'[{"accession":"GCA_900706755.1","sample_accession":"SAMEA1","scientific_name":"Escherichia coli","strain":"NCTC9001","tax_id":"562"}]'
    asms = msgspec.json.decode(asm_json, type=list[ENAAssemblyRecord])
    assert len(asms) == 1
    assert asms[0].accession == "GCA_900706755.1"

    nuc = NuccoreSummary(uid="123", accession_version="AB811596.1", title="E. coli O1", slen=12000)
    pm = PubMedSummary(uid="15247885", title="Test", doi="10.1000/1")
    assert nuc.accession_version == "AB811596.1"
    assert pm.doi == "10.1000/1"


def test_client_instantiation_and_pathogenwatch_collection() -> None:
    c = Client("https://example.org/api/", api_key="secret", api_key_header="X-Key")
    assert c.base_url == "https://example.org/api"
    assert "**********" in repr(c)

    ncbi = NCBIClient(api_key="testkey")
    ena = ENAClient()
    si = StrainInfoClient()
    bd = BacDiveClient(api_key="bdkey")
    pw = PathogenwatchClient(api_key="pwkey")
    assert ncbi.api_key == "testkey"
    assert ena.base_url.endswith("/ena/portal/api")
    assert si.base_url.endswith("/v2")
    assert bd.api_key == "Bearer bdkey"
    assert pw.api_key == "pwkey"

    col = PathogenwatchCollection.from_dict({"uuid": "abc", "name": "Set 1", "genomeCount": 42, "organismId": "573"})
    assert col.uuid == "abc"
    assert col.size == 42
    assert col.organism_name == "Klebsiella pneumoniae"
