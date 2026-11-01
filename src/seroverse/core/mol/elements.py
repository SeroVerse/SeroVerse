"""Periodic table of chemical elements (`Element`, `ElementSpec`, `ElementSpecs`)."""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from seroverse.core.collections import Spec, SpecEnum, Specs
from seroverse.core.types import Types


@dataclass(frozen=True, slots=True)
class ElementSpec(Spec):
    r"""Immutable scalar specification for a chemical element in the periodic table.

    Attributes:
        name: Canonical IUPAC atomic symbol as ASCII `bytes` (e.g. `b"C"`, `b"Fe"`, `b"*"`).
        full_name: Full lowercase IUPAC element name as ASCII `bytes` (e.g. `b"carbon"`, `b"iron"`).
        atomic_number: Atomic number $Z \in [0, 118]$ (`0` for wildcard `*`).
        mass: Standard atomic weight (or most stable isotope mass) in Daltons (Da).
        default_valence: Primary neutral covalent valence used for implicit hydrogen inference.
        electronegativity: Pauling electronegativity ($\chi$).
        covalent_radius: Single-bond covalent radius in Ångströms (Å).
        vdw_radius: Van der Waals radius in Ångströms (Å).
    """

    full_name: bytes
    atomic_number: int
    mass: float
    default_valence: int = 0
    electronegativity: float = 0.0
    covalent_radius: float = 1.50
    vdw_radius: float = 2.00

    @property
    def full_name_str(self) -> str:
        r"""Full IUPAC element name decoded as a UTF-8 `str`."""
        return self.full_name.decode("utf-8", errors="replace")

    def lookup_keys(self) -> tuple[bytes, ...]:
        r"""Register both the atomic symbol and the full IUPAC element name for $O(1)$ lookup."""
        return (self.name.strip().lower(), self.full_name.strip().lower())


@dataclass(frozen=True, slots=True)
class ElementSpecs(Specs[ElementSpec]):
    r"""Compiled Structure-of-Arrays (SoA) periodic table specification container for [`Element`][seroverse.core.mol.elements.Element]."""

    _scalar_type: ClassVar[type[Any] | None] = ElementSpec

    full_names: Types.BytesArray
    atomic_numbers: Types.UInt8Array
    masses: Types.Float32Array
    default_valences: Types.UInt8Array
    electronegativities: Types.Float32Array
    covalent_radii: Types.Float32Array
    vdw_radii: Types.Float32Array

    @property
    def feature_matrix(self) -> Types.Float32Array2D:
        r"""Normalized `(119, 6)` `float32` physicochemical feature matrix across all elements."""
        n = len(self.names)
        features = np.empty((n, 6), dtype=np.float32)
        features[:, 0] = self.atomic_numbers / np.float32(118.0)
        features[:, 1] = self.masses / np.float32(294.0)
        features[:, 2] = self.default_valences / np.float32(8.0)
        features[:, 3] = self.electronegativities / np.float32(4.0)
        features[:, 4] = self.covalent_radii / np.float32(2.6)
        features[:, 5] = self.vdw_radii / np.float32(3.5)
        features.setflags(write=False)
        return features


def _el(
    z: int,
    sym: bytes,
    full_name: bytes,
    mass: float,
    valence: int,
    en: float,
    r_cov: float,
    r_vdw: float,
) -> ElementSpec:
    return ElementSpec(
        name=sym,
        full_name=full_name,
        atomic_number=z,
        mass=mass,
        default_valence=valence,
        electronegativity=en,
        covalent_radius=r_cov,
        vdw_radius=r_vdw,
    )


_COMPILED_ELEMENT_SPECS: ElementSpecs | None = None


class Element(SpecEnum):
    r"""IUPAC periodic table of chemical elements (`0..118`, with `0 = WILDCARD / *`)."""

    WILDCARD = 0
    H = 1
    HE = 2
    LI = 3
    BE = 4
    B = 5
    C = 6
    N = 7
    O = 8  # noqa: E741
    F = 9
    NE = 10
    NA = 11
    MG = 12
    AL = 13
    SI = 14
    P = 15
    S = 16
    CL = 17
    AR = 18
    K = 19
    CA = 20
    SC = 21
    TI = 22
    V = 23
    CR = 24
    MN = 25
    FE = 26
    CO = 27
    NI = 28
    CU = 29
    ZN = 30
    GA = 31
    GE = 32
    AS = 33
    SE = 34
    BR = 35
    KR = 36
    RB = 37
    SR = 38
    Y = 39
    ZR = 40
    NB = 41
    MO = 42
    TC = 43
    RU = 44
    RH = 45
    PD = 46
    AG = 47
    CD = 48
    IN = 49
    SN = 50
    SB = 51
    TE = 52
    I = 53  # noqa: E741
    XE = 54
    CS = 55
    BA = 56
    LA = 57
    CE = 58
    PR = 59
    ND = 60
    PM = 61
    SM = 62
    EU = 63
    GD = 64
    TB = 65
    DY = 66
    HO = 67
    ER = 68
    TM = 69
    YB = 70
    LU = 71
    HF = 72
    TA = 73
    W = 74
    RE = 75
    OS = 76
    IR = 77
    PT = 78
    AU = 79
    HG = 80
    TL = 81
    PB = 82
    BI = 83
    PO = 84
    AT = 85
    RN = 86
    FR = 87
    RA = 88
    AC = 89
    TH = 90
    PA = 91
    U = 92
    NP = 93
    PU = 94
    AM = 95
    CM = 96
    BK = 97
    CF = 98
    ES = 99
    FM = 100
    MD = 101
    NO = 102
    LR = 103
    RF = 104
    DB = 105
    SG = 106
    BH = 107
    HS = 108
    MT = 109
    DS = 110
    RG = 111
    CN = 112
    NH = 113
    FL = 114
    MC = 115
    LV = 116
    TS = 117
    OG = 118

    @classmethod
    def _compile_specs(cls) -> ElementSpecs:
        return ElementSpecs.from_list(
            [
                _el(0, b"*", b"wildcard", 0.0, 0, 0.0, 0.0, 0.0),
                _el(1, b"H", b"hydrogen", 1.008, 1, 2.20, 0.31, 1.20),
                _el(2, b"He", b"helium", 4.0026, 0, 0.0, 0.28, 1.40),
                _el(3, b"Li", b"lithium", 6.94, 1, 0.98, 1.28, 1.82),
                _el(4, b"Be", b"beryllium", 9.0122, 2, 1.57, 0.96, 1.53),
                _el(5, b"B", b"boron", 10.81, 3, 2.04, 0.84, 1.92),
                _el(6, b"C", b"carbon", 12.011, 4, 2.55, 0.76, 1.70),
                _el(7, b"N", b"nitrogen", 14.007, 3, 3.04, 0.71, 1.55),
                _el(8, b"O", b"oxygen", 15.999, 2, 3.44, 0.66, 1.52),
                _el(9, b"F", b"fluorine", 18.998, 1, 3.98, 0.57, 1.47),
                _el(10, b"Ne", b"neon", 20.180, 0, 0.0, 0.58, 1.54),
                _el(11, b"Na", b"sodium", 22.990, 1, 0.93, 1.66, 2.27),
                _el(12, b"Mg", b"magnesium", 24.305, 2, 1.31, 1.41, 1.73),
                _el(13, b"Al", b"aluminium", 26.982, 3, 1.61, 1.21, 1.84),
                _el(14, b"Si", b"silicon", 28.085, 4, 1.90, 1.11, 2.10),
                _el(15, b"P", b"phosphorus", 30.974, 3, 2.19, 1.07, 1.80),
                _el(16, b"S", b"sulfur", 32.06, 2, 2.58, 1.05, 1.80),
                _el(17, b"Cl", b"chlorine", 35.45, 1, 3.16, 1.02, 1.75),
                _el(18, b"Ar", b"argon", 39.948, 0, 0.0, 1.06, 1.88),
                _el(19, b"K", b"potassium", 39.098, 1, 0.82, 2.03, 2.75),
                _el(20, b"Ca", b"calcium", 40.078, 2, 1.00, 1.76, 2.31),
                _el(21, b"Sc", b"scandium", 44.956, 3, 1.36, 1.70, 2.11),
                _el(22, b"Ti", b"titanium", 47.867, 4, 1.54, 1.60, 2.00),
                _el(23, b"V", b"vanadium", 50.942, 5, 1.63, 1.53, 2.00),
                _el(24, b"Cr", b"chromium", 51.996, 6, 1.66, 1.39, 2.00),
                _el(25, b"Mn", b"manganese", 54.938, 2, 1.55, 1.39, 2.00),
                _el(26, b"Fe", b"iron", 55.845, 2, 1.83, 1.32, 2.00),
                _el(27, b"Co", b"cobalt", 58.933, 2, 1.88, 1.26, 2.00),
                _el(28, b"Ni", b"nickel", 58.693, 2, 1.91, 1.24, 1.63),
                _el(29, b"Cu", b"copper", 63.546, 2, 1.90, 1.32, 1.40),
                _el(30, b"Zn", b"zinc", 65.38, 2, 1.65, 1.22, 1.39),
                _el(31, b"Ga", b"gallium", 69.723, 3, 1.81, 1.22, 1.87),
                _el(32, b"Ge", b"germanium", 72.630, 4, 2.01, 1.20, 2.11),
                _el(33, b"As", b"arsenic", 74.922, 3, 2.18, 1.19, 1.85),
                _el(34, b"Se", b"selenium", 78.971, 2, 2.55, 1.20, 1.90),
                _el(35, b"Br", b"bromine", 79.904, 1, 2.96, 1.20, 1.85),
                _el(36, b"Kr", b"krypton", 83.798, 0, 3.00, 1.16, 2.02),
                _el(37, b"Rb", b"rubidium", 85.468, 1, 0.82, 2.20, 3.03),
                _el(38, b"Sr", b"strontium", 87.62, 2, 0.95, 1.95, 2.49),
                _el(39, b"Y", b"yttrium", 88.906, 3, 1.22, 1.90, 2.32),
                _el(40, b"Zr", b"zirconium", 91.224, 4, 1.33, 1.75, 2.23),
                _el(41, b"Nb", b"niobium", 92.906, 5, 1.60, 1.64, 2.18),
                _el(42, b"Mo", b"molybdenum", 95.95, 6, 2.16, 1.54, 2.17),
                _el(43, b"Tc", b"technetium", 98.0, 7, 1.90, 1.47, 2.16),
                _el(44, b"Ru", b"ruthenium", 101.07, 4, 2.20, 1.46, 2.13),
                _el(45, b"Rh", b"rhodium", 102.91, 3, 2.28, 1.42, 2.10),
                _el(46, b"Pd", b"palladium", 106.42, 2, 2.20, 1.39, 1.63),
                _el(47, b"Ag", b"silver", 107.87, 1, 1.93, 1.45, 1.72),
                _el(48, b"Cd", b"cadmium", 112.41, 2, 1.69, 1.44, 1.58),
                _el(49, b"In", b"indium", 114.82, 3, 1.78, 1.42, 1.93),
                _el(50, b"Sn", b"tin", 118.71, 4, 1.96, 1.39, 2.17),
                _el(51, b"Sb", b"antimony", 121.76, 3, 2.05, 1.39, 2.06),
                _el(52, b"Te", b"tellurium", 127.60, 2, 2.10, 1.38, 2.06),
                _el(53, b"I", b"iodine", 126.90, 1, 2.66, 1.39, 1.98),
                _el(54, b"Xe", b"xenon", 131.29, 0, 2.60, 1.40, 2.16),
                _el(55, b"Cs", b"cesium", 132.91, 1, 0.79, 2.44, 3.43),
                _el(56, b"Ba", b"barium", 137.33, 2, 0.89, 2.15, 2.68),
                _el(57, b"La", b"lanthanum", 138.91, 3, 1.10, 2.07, 2.43),
                _el(58, b"Ce", b"cerium", 140.12, 3, 1.12, 2.04, 2.42),
                _el(59, b"Pr", b"praseodymium", 140.91, 3, 1.13, 2.03, 2.40),
                _el(60, b"Nd", b"neodymium", 144.24, 3, 1.14, 2.01, 2.39),
                _el(61, b"Pm", b"promethium", 145.0, 3, 1.13, 1.99, 2.38),
                _el(62, b"Sm", b"samarium", 150.36, 3, 1.17, 1.98, 2.36),
                _el(63, b"Eu", b"europium", 151.96, 3, 1.20, 1.98, 2.35),
                _el(64, b"Gd", b"gadolinium", 157.25, 3, 1.20, 1.96, 2.34),
                _el(65, b"Tb", b"terbium", 158.93, 3, 1.20, 1.94, 2.33),
                _el(66, b"Dy", b"dysprosium", 162.50, 3, 1.22, 1.92, 2.31),
                _el(67, b"Ho", b"holmium", 164.93, 3, 1.23, 1.92, 2.30),
                _el(68, b"Er", b"erbium", 167.26, 3, 1.24, 1.89, 2.29),
                _el(69, b"Tm", b"thulium", 168.93, 3, 1.25, 1.90, 2.27),
                _el(70, b"Yb", b"ytterbium", 173.05, 3, 1.10, 1.87, 2.26),
                _el(71, b"Lu", b"lutetium", 174.97, 3, 1.27, 1.87, 2.24),
                _el(72, b"Hf", b"hafnium", 178.49, 4, 1.30, 1.75, 2.23),
                _el(73, b"Ta", b"tantalum", 180.95, 5, 1.50, 1.70, 2.22),
                _el(74, b"W", b"tungsten", 183.84, 6, 2.36, 1.62, 2.18),
                _el(75, b"Re", b"rhenium", 186.21, 7, 1.90, 1.51, 2.16),
                _el(76, b"Os", b"osmium", 190.23, 4, 2.20, 1.44, 2.16),
                _el(77, b"Ir", b"iridium", 192.22, 4, 2.20, 1.41, 2.13),
                _el(78, b"Pt", b"platinum", 195.08, 2, 2.28, 1.36, 1.75),
                _el(79, b"Au", b"gold", 196.97, 1, 2.54, 1.36, 1.66),
                _el(80, b"Hg", b"mercury", 200.59, 2, 2.00, 1.32, 1.55),
                _el(81, b"Tl", b"thallium", 204.38, 3, 1.62, 1.45, 1.96),
                _el(82, b"Pb", b"lead", 207.2, 4, 2.33, 1.46, 2.02),
                _el(83, b"Bi", b"bismuth", 208.98, 3, 2.02, 1.48, 2.07),
                _el(84, b"Po", b"polonium", 209.0, 2, 2.00, 1.40, 1.97),
                _el(85, b"At", b"astatine", 210.0, 1, 2.20, 1.50, 2.02),
                _el(86, b"Rn", b"radon", 222.0, 0, 0.0, 1.50, 2.20),
                _el(87, b"Fr", b"francium", 223.0, 1, 0.70, 2.60, 3.48),
                _el(88, b"Ra", b"radium", 226.0, 2, 0.90, 2.21, 2.83),
                _el(89, b"Ac", b"actinium", 227.0, 3, 1.10, 2.15, 2.47),
                _el(90, b"Th", b"thorium", 232.04, 4, 1.30, 2.06, 2.45),
                _el(91, b"Pa", b"protactinium", 231.04, 5, 1.50, 2.00, 2.43),
                _el(92, b"U", b"uranium", 238.03, 6, 1.38, 1.96, 1.86),
                _el(93, b"Np", b"neptunium", 237.0, 5, 1.36, 1.90, 2.39),
                _el(94, b"Pu", b"plutonium", 244.0, 4, 1.28, 1.87, 2.43),
                _el(95, b"Am", b"americium", 243.0, 3, 1.30, 1.80, 2.44),
                _el(96, b"Cm", b"curium", 247.0, 3, 1.30, 1.69, 2.45),
                _el(97, b"Bk", b"berkelium", 247.0, 3, 1.30, 1.68, 2.44),
                _el(98, b"Cf", b"californium", 251.0, 3, 1.30, 1.68, 2.45),
                _el(99, b"Es", b"einsteinium", 252.0, 3, 1.30, 1.65, 2.45),
                _el(100, b"Fm", b"fermium", 257.0, 3, 1.30, 1.67, 2.45),
                _el(101, b"Md", b"mendelevium", 258.0, 3, 1.30, 1.73, 2.46),
                _el(102, b"No", b"nobelium", 259.0, 2, 1.30, 1.76, 2.46),
                _el(103, b"Lr", b"lawrencium", 266.0, 3, 1.30, 1.61, 2.46),
                _el(104, b"Rf", b"rutherfordium", 267.0, 4, 0.0, 1.57, 2.30),
                _el(105, b"Db", b"dubnium", 268.0, 5, 0.0, 1.49, 2.30),
                _el(106, b"Sg", b"seaborgium", 269.0, 6, 0.0, 1.43, 2.30),
                _el(107, b"Bh", b"bohrium", 270.0, 7, 0.0, 1.41, 2.30),
                _el(108, b"Hs", b"hassium", 277.0, 8, 0.0, 1.34, 2.30),
                _el(109, b"Mt", b"meitnerium", 278.0, 0, 0.0, 1.29, 2.30),
                _el(110, b"Ds", b"darmstadtium", 281.0, 0, 0.0, 1.28, 2.30),
                _el(111, b"Rg", b"roentgenium", 282.0, 0, 0.0, 1.21, 2.30),
                _el(112, b"Cn", b"copernicium", 285.0, 2, 0.0, 1.22, 2.30),
                _el(113, b"Nh", b"nihonium", 286.0, 3, 0.0, 1.36, 2.30),
                _el(114, b"Fl", b"flerovium", 289.0, 4, 0.0, 1.43, 2.30),
                _el(115, b"Mc", b"moscovium", 290.0, 3, 0.0, 1.62, 2.30),
                _el(116, b"Lv", b"livermorium", 293.0, 2, 0.0, 1.75, 2.30),
                _el(117, b"Ts", b"tennessine", 294.0, 1, 0.0, 1.65, 2.30),
                _el(118, b"Og", b"oganesson", 294.0, 0, 0.0, 1.57, 2.30),
            ]
        )

    @classmethod
    def specs(cls) -> ElementSpecs:
        r"""Return the cached, validated singleton [`ElementSpecs`][seroverse.core.mol.elements.ElementSpecs] table."""
        global _COMPILED_ELEMENT_SPECS
        if _COMPILED_ELEMENT_SPECS is None:
            super().specs()
            _COMPILED_ELEMENT_SPECS = cls._compile_specs()
        return _COMPILED_ELEMENT_SPECS

    @property
    def spec(self) -> ElementSpec:
        r"""Return the scalar [`ElementSpec`][seroverse.core.mol.elements.ElementSpec] view for this element."""
        return self.specs()[int(self)]
