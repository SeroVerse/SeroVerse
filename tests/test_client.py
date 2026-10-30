"""Unit tests for `seroverse.client` models, `pyreqwest` clients, `ClientMocker`, and `AsyncClientProxy`."""

import asyncio
import gzip
from typing import Any

import pytest
from pyreqwest.exceptions import StatusError
from pyreqwest.pytest_plugin import ClientMocker

from seroverse.client.atcc import ATCCAssembly, ATCCGenome, ATCCGenomePortalClient
from seroverse.client.base import AsyncClientProxy, Client
from seroverse.client.dsmz import BacDiveClient, StrainInfoClient, StrainPassport
from seroverse.client.ena import ENAAssemblyRecord, ENAClient, ENASampleRecord
from seroverse.client.ncbi import NCBIClient, NuccoreSummary, PubMedSummary
from seroverse.client.pathogenwatch import PathogenwatchClient, PathogenwatchCollection
from seroverse.client.uniprot import UniProtClient, UniProtEntry


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


def test_ena_and_ncbi_clients_with_client_mocker(monkeypatch: pytest.MonkeyPatch) -> None:
    mocker = ClientMocker.create_mocker(monkeypatch).strict()

    sample_payload = [
        {
            "sample_accession": "SAMEA1",
            "scientific_name": "Escherichia coli",
            "strain": "NCTC9001",
            "serotype": "O1:K1:H7",
            "serovar": "",
            "tax_id": "562",
        }
    ]
    asm_payload = [
        {
            "accession": "GCA_900706755.1",
            "sample_accession": "SAMEA1",
            "scientific_name": "Escherichia coli",
            "strain": "NCTC9001",
            "tax_id": "562",
        }
    ]

    ena_sample_mock = (
        mocker.get(path="/ena/portal/api/search").match_query_param("result", "sample").with_body_json(sample_payload)
    )
    ena_asm_mock = (
        mocker.get(path="/ena/portal/api/search").match_query_param("result", "assembly").with_body_json(asm_payload)
    )

    with ENAClient(polite_delay=0.0) as ena:
        samples = ena.search_samples("tax_eq(562)", limit=5)
        assert len(samples) == 1
        assert samples[0] == ENASampleRecord(
            sample_accession="SAMEA1",
            scientific_name="Escherichia coli",
            strain="NCTC9001",
            serotype="O1:K1:H7",
            serovar="",
            tax_id="562",
        )
        asms = ena.search_assemblies("tax_eq(562)", limit=5)
        assert len(asms) == 1
        assert asms[0] == ENAAssemblyRecord(
            accession="GCA_900706755.1",
            sample_accession="SAMEA1",
            scientific_name="Escherichia coli",
            strain="NCTC9001",
            tax_id="562",
        )
    ena_sample_mock.assert_called()
    ena_asm_mock.assert_called()

    esearch_mock = (
        mocker.get(path="/entrez/eutils/esearch.fcgi")
        .match_query_param("db", "nuccore")
        .match_query_param("api_key", "ncbi-test-key")
        .with_body_json({"esearchresult": {"idlist": ["123", "456"]}})
    )
    esummary_mock = mocker.post(path="/entrez/eutils/esummary.fcgi").with_body_json(
        {
            "result": {
                "123": {
                    "accessionversion": "AB811596.1",
                    "title": "E. coli O1",
                    "subname": "O1",
                    "subtype": "serotype",
                    "slen": 12000,
                },
                "456": {
                    "accessionversion": "AB000001.1",
                    "title": "Too short",
                    "slen": 500,
                },
            }
        }
    )
    elink_mock = (
        mocker.get(path="/entrez/eutils/elink.fcgi")
        .match_query_param("id", ["15247885", "15247886"])
        .with_body_json(
            {
                "linksets": [
                    {
                        "ids": [15247885],
                        "linksetdbs": [{"linkname": "pubmed_nuccore", "links": [123]}],
                    }
                ]
            }
        )
    )

    with NCBIClient(api_key="ncbi-test-key", polite_delay=0.0) as ncbi:
        uids = ncbi.esearch("nuccore", "Escherichia coli")
        assert uids == ["123", "456"]
        summaries = ncbi.fetch_nuccore_summaries(uids, slen_range=(5000, 20000))
        assert list(summaries) == ["123"]
        assert summaries["123"].base_accession == "AB811596"
        assert summaries["123"].sub_pairs == [("serotype", "O1")]
        linked = ncbi.elink_pubmed_to_nuccore([15247885, 15247886])
        assert linked == {15247885: ["123"]}

    esearch_mock.assert_called()
    esummary_mock.assert_called()
    elink_mock.assert_called()

    pm = PubMedSummary(uid="15247885", title="Test", doi="10.1000/1")
    assert NuccoreSummary(uid="123", accession_version="AB811596.1").accession_version == "AB811596.1"
    assert pm.doi == "10.1000/1"


def test_client_instantiation_and_dsmz_pathogenwatch_mocker(monkeypatch: pytest.MonkeyPatch) -> None:
    mocker = ClientMocker.create_mocker(monkeypatch).strict()

    mocker.get(path="/api/ping").match_header("X-Key", "secret").with_body_json({"ok": True})
    mocker.get(path="/api/missing").with_status(404).with_body_text("Not Found")

    with Client("https://example.org/api/", api_key="secret", api_key_header="X-Key") as c:
        assert c.base_url == "https://example.org/api"
        assert "**********" in repr(c)
        assert c.headers["X-Key"] == "secret"
        assert c.get_data("ping") == {"ok": True}
        assert c.get_data("missing", allow_404=True) is None
        assert c.stream_bytes("GET", "missing", allow_404=True) == b""
        with pytest.raises(StatusError) as exc_info:
            c.get_data("missing")
        assert exc_info.value.details["status"] == 404

    mocker.get(path="/v2/search/strain/tax_name/Proteus%20mirabilis").with_body_json([108168])
    mocker.get(path="/v2/data/strain/max/108168").with_body_json(
        [{"strain": {"id": 108168, "taxon": {"name": "Proteus mirabilis", "ncbi": 584}}}]
    )
    mocker.get(path="/v2/fetch/143413").match_header("Authorization", "Bearer bdkey").with_body_json(
        {"General": {"BacDive-ID": 143413}}
    )
    mocker.get(path="/api/collections/list").match_header("X-API-Key", "pwkey").with_body_json(
        [{"uuid": "abc", "name": "Set 1", "genomeCount": 42, "organismId": "573"}]
    )

    with (
        StrainInfoClient(polite_delay=0.0) as si,
        BacDiveClient(api_key="bdkey", polite_delay=0.0) as bd,
        PathogenwatchClient(api_key="pwkey", polite_delay=0.0) as pw,
    ):
        assert si.base_url.endswith("/v2")
        assert bd.api_key == "Bearer bdkey"
        assert pw.api_key == "pwkey"

        passports = si.fetch_taxon_passports("Proteus mirabilis")
        assert len(passports) == 1
        assert passports[0].si_id == 108168

        strain = bd.fetch_strain(143413)
        assert strain == {"General": {"BacDive-ID": 143413}}

        cols = pw.get_collections()
        assert len(cols) == 1
        assert cols[0] == PathogenwatchCollection(
            uuid="abc",
            name="Set 1",
            size=42,
            organism_id="573",
            organism_name="Klebsiella pneumoniae",
        )


def test_atcc_genome_portal_models_and_sync_plus_async_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    mocker = ClientMocker.create_mocker(monkeypatch).strict()

    with pytest.raises(ValueError, match="ATCC Genome Portal API key"):
        ATCCGenomePortalClient(api_key="   ")

    client = ATCCGenomePortalClient(
        api_key="atcc-secret-key",
        base_url="https://genomes.atcc.org",
        polite_delay=0.0,
    )
    assert client.base_url == "https://genomes.atcc.org/api"
    assert client.api_key == "atcc-secret-key"
    assert client.api_key_header == "X-API-Key"
    assert client.headers["X-API-Key"] == "atcc-secret-key"

    bearer_client = ATCCGenomePortalClient(api_key="Bearer token-123", polite_delay=0.0)
    assert bearer_client.api_key_header == "Authorization"
    bearer_client.close()

    raw_genome: dict[str, Any] = {
        "id": "ab12cd34",
        "product_id": "ATCC BAA-1716",
        "taxon_name": "Klebsiella pneumoniae",
        "ncbi_taxid": "573",
        "atcc_metadata": {
            "strain_designations": ["KP1", "NCTC 9633"],
            "serotype": "K1",
            "isolation_source": "Blood",
            "isolation_country": "USA",
            "collection_year": 2015,
        },
        "primary_assembly": {
            "id": "asm-001",
            "version": "1.0",
            "assembly_level": "Complete Genome",
            "assembly_statistics": {
                "total_length": 5432100,
                "n_contigs": 2,
                "n50": 5200000,
                "gc_content": 57.3,
                "sequencing_technology": "Illumina + ONT",
            },
        },
    }
    genome = ATCCGenome.from_dict(raw_genome)
    assert genome.id == "ab12cd34"
    assert genome.product_id == "BAA-1716"
    assert genome.catalog_label == "ATCC BAA-1716"
    assert genome.taxon_name == "Klebsiella pneumoniae"
    assert genome.ncbi_taxid == 573
    assert genome.strain == "KP1, NCTC 9633"
    assert genome.serotype == "K1"
    assert genome.isolation_source == "Blood"
    assert genome.geographic_location == "USA"
    assert genome.collection_date == "2015"
    assert genome.primary_assembly_id == "asm-001"
    assert genome.primary_assembly is not None
    assert genome.primary_assembly.genome_id == "ab12cd34"
    assert genome.primary_assembly.genome_size == 5432100
    assert genome.primary_assembly.n_contigs == 2
    assert genome.primary_assembly.n50 == 5200000
    assert genome.primary_assembly.gc_content == pytest.approx(57.3)
    assert genome.primary_assembly.sequencing_technology == "Illumina + ONT"

    asm_direct = ATCCAssembly.from_dict({"id": "asm-001", "genome_id": "ab12cd34", "total_length": 5432100})
    assert asm_direct.id == "asm-001"
    assert asm_direct.genome_size == 5432100

    fasta_payload = b">contig1 description\nACGTACGT\n>contig2\nTTTTGGGG\n"
    gz_fasta = gzip.compress(fasta_payload)

    search_mock = (
        mocker.post(path="/api/genomes/search")
        .match_header("X-API-Key", "atcc-secret-key")
        .match_body_json({"product_id": "BAA-1716"})
        .with_body_json({"results": [raw_genome], "has_next": False})
    )
    dl_asm_mock = (
        mocker.get(path="/api/genomes/ab12cd34/download_assembly")
        .match_header("X-API-Key", "atcc-secret-key")
        .with_body_json({"url": "https://s3.example.com/signed/assembly.fna.gz"})
    )
    s3_mock = (
        mocker.get(url="https://s3.example.com/signed/assembly.fna.gz")
        .match_request(lambda req: req.headers.get("X-API-Key") is None)
        .with_body_bytes(gz_fasta)
    )
    dl_annot_mock = (
        mocker.get(path="/api/genomes/ab12cd34/download_annotations")
        .match_header("X-API-Key", "atcc-secret-key")
        .with_body_bytes(b"##gff-version 3\n")
    )

    # 1. Synchronous usage
    results = client.search_by_product_id("ATCC BAA-1716")
    assert len(results) == 1
    assert results[0].product_id == "BAA-1716"

    asm_bytes = client.download_assembly("ab12cd34")
    assert asm_bytes == fasta_payload

    raw_gz = client.download_assembly("ab12cd34", decompress=False)
    assert raw_gz == gz_fasta

    annot_bytes = client.download_annotations("ab12cd34")
    assert annot_bytes == b"##gff-version 3\n"

    seqs = client.fetch_assembly_sequences("ab12cd34")
    assert len(seqs) == 2
    assert bytes(seqs[0].id) == b"contig1"
    assert bytes(seqs[0].seq) == b"ACGTACGT"
    assert bytes(seqs[1].id) == b"contig2"
    assert bytes(seqs[1].seq) == b"TTTTGGGG"

    # 2. Asynchronous proxy usage via `async with client:`
    async def _run_async() -> None:
        async with client as aclient:
            assert isinstance(aclient, AsyncClientProxy)
            assert aclient.__cache_fingerprint__() == client.__cache_fingerprint__()
            bound = aclient.search_by_product_id
            assert getattr(bound, "__self__", None) is client
            async_res = await aclient.search_by_product_id("ATCC BAA-1716")
            assert len(async_res) == 1
            assert async_res[0].id == "ab12cd34"
            async_seqs = await aclient.fetch_assembly_sequences("ab12cd34")
            assert len(async_seqs) == 2

    asyncio.run(_run_async())

    search_mock.assert_called(count=2)
    dl_asm_mock.assert_called(count=4)
    s3_mock.assert_called(count=4)
    dl_annot_mock.assert_called(count=1)


def test_uniprot_client_streaming_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    mocker = ClientMocker.create_mocker(monkeypatch).strict()
    client = UniProtClient(polite_delay=0.0)
    assert client.base_url == "https://rest.uniprot.org"

    fasta_payload = (
        b">sp|P12345|WZX_ECOLI Lipopolysaccharide flippase OS=Escherichia coli OX=562 GN=wzx\n"
        b"MKTLLILAVV\n"
        b">tr|Q99999|GTRB_BP123 Bacteriophage glucosyltransferase OS=Klebsiella phage KP1 OX=10239\n"
        b"ACDEFGHIKL\n"
        b">tr|A0A001|WZY_KLEPN O-antigen polymerase OS=Klebsiella pneumoniae OX=573\n"
        b"WYWYWYWYWY\n"
    )
    gz_fasta = gzip.compress(fasta_payload)

    fasta_search_mock = (
        mocker.get(path="/uniprotkb/search")
        .match_query(
            {
                "query": "(gene:wzx AND taxonomy_id:2) AND fragment:false",
                "format": "fasta",
                "size": "150",
            }
        )
        .match_header("Accept", "text/plain; format=fasta")
        .with_body_bytes(fasta_payload)
    )
    uniref_stream_mock = (
        mocker.get(path="/uniref/stream")
        .match_query(
            {
                "query": "uniref_cluster_50:wzx",
                "format": "fasta",
                "compressed": "true",
            }
        )
        .match_header("Accept", "text/plain; format=fasta")
        .with_body_bytes(gz_fasta)
    )

    # 1. `limit <= 500` uses `/uniprotkb/search` with `AND fragment:false`
    seqs = client.fetch_sequences("gene:wzx AND taxonomy_id:2", limit=150, truncate_id=False)
    assert len(seqs) == 3
    assert b"OS=Klebsiella phage KP1" in bytes(seqs[1].id)
    assert bytes(seqs[0].seq) == b"MKTLLILAVV"
    fasta_search_mock.assert_called(count=1)

    # 2. `limit > 500` or `limit=None` uses `/uniref/stream` (and does not add `fragment:false` to uniref)
    streamed = list(
        client.stream_sequences(
            "uniref_cluster_50:wzx",
            database="uniref",
            limit=None,
            compressed=True,
            truncate_id=True,
        )
    )
    assert len(streamed) == 3
    assert bytes(streamed[0].id) == b"sp|P12345|WZX_ECOLI"
    assert bytes(streamed[1].id) == b"tr|Q99999|GTRB_BP123"
    uniref_stream_mock.assert_called(count=1)

    # 3. `UniProtEntry.from_dict` and `get_entry` / `search` / 404
    raw_entry = {
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
        "primaryAccession": "P12345",
        "uniProtkbId": "WZX_ECOLI",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Lipopolysaccharide flippase"}}},
        "genes": [{"geneName": {"value": "wzx"}, "synonyms": [{"value": "rfbX"}]}],
        "organism": {"scientificName": "Escherichia coli", "taxonId": 562},
        "sequence": {"value": "MKTLLILAVV", "length": 10},
    }
    entry_mock = mocker.get(path="/uniprotkb/P12345.json").with_body_json(raw_entry)
    missing_entry_mock = mocker.get(path="/uniprotkb/MISSING.json").with_status(404)
    json_search_mock = (
        mocker.get(path="/uniprotkb/search")
        .match_query(
            {
                "query": "(gene:wzx) AND fragment:false",
                "format": "json",
                "size": "10",
            }
        )
        .with_body_json({"results": [raw_entry]})
    )

    entry = client.get_entry("P12345")
    assert isinstance(entry, UniProtEntry)
    assert entry.accession == "P12345"
    assert entry.id == "WZX_ECOLI"
    assert entry.protein_name == "Lipopolysaccharide flippase"
    assert entry.gene_names == ("wzx", "rfbX")
    assert entry.organism_name == "Escherichia coli"
    assert entry.organism_id == 562
    assert entry.length == 10
    assert entry.reviewed is True
    assert entry.is_fragment is False
    assert entry.sequence == "MKTLLILAVV"
    entry_mock.assert_called(count=1)

    assert client.get_entry("MISSING") is None
    missing_entry_mock.assert_called(count=1)

    search_hits = client.search("gene:wzx", limit=10)
    assert len(search_hits) == 1
    assert search_hits[0] == entry
    json_search_mock.assert_called(count=1)
    client.close()
