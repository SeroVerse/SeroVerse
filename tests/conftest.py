"""Shared pytest fixtures for SeroVerse genome assembly FASTA paths."""

from pathlib import Path

import pytest

DATA_DIR: Path = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------------------------------------------------------
# Raw Genome Assembly Path Fixtures
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pao1_path() -> Path:
    """Path to the zstd-compressed *Pseudomonas aeruginosa* PAO1 genome assembly."""
    return DATA_DIR / "PAO1.fna.zst"


@pytest.fixture(scope="session")
def sgh10_path() -> Path:
    """Path to the zstd-compressed *Klebsiella pneumoniae* SGH10 chromosome assembly."""
    return DATA_DIR / "SGH10_chromosome.fna.zst"


@pytest.fixture(scope="session")
def stmc125_path() -> Path:
    """Path to the zstd-compressed *Salmonella enterica* STMC125 genome assembly."""
    return DATA_DIR / "STMC125.fna.zst"


@pytest.fixture(
    scope="session",
    params=("pao1_path", "sgh10_path", "stmc125_path"),
    ids=("PAO1", "SGH10_chromosome", "STMC125"),
)
def assembly_path(request: pytest.FixtureRequest) -> Path:
    """Parametrized fixture yielding each test genome assembly file path."""
    path: Path = request.getfixturevalue(request.param)
    return path
