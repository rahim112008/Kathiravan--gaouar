"""config_nw.py — Configuration des camélidés du Nouveau Monde"""
from __future__ import annotations

REFERENCE = {
    "name":       "VicPac4",
    "accession":  "GCF_048564905.1",
    "species":    "Vicugna pacos",
    "ncbi_url":   "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_048564905.1/",
}

ARS_VICPAC4 = {str(i): 90.0 for i in range(1, 37)} | {"X": 140.0, "Y": 45.0, "MT": 16.6}
AUTOSOMES   = [str(i) for i in range(1, 37)]
SEX_CHR     = ["X", "Y"]
MT          = ["MT"]

SPECIES_CONFIG = {
    "🦙 Alpaga (Vicugna pacos)": {
        "taxon": "30538", "vep_species": "vicugna_pacos", "prefix": "alpaca",
        "karyotype": "2n = 74",
        "groupes": {
            "Race": {
                "HUACAYA_BOL":   ["BAH"],
                "SURI_BOL":      ["BAS"],
                "HUACAYA_PER":   ["PAH"],
                "SURI_PER":      ["PAS"],
                "US_ALPACA":     ["ULP"],
            },
            "Pays": {
                "BOLIVIE": ["BAH", "BAS"],
                "PEROU":   ["PAH", "PAS"],
                "USA":     ["ULP"],
            },
        },
    },
    "🦌 Guanaco (Lama guanicoe)": {
        "taxon": "9840", "vep_species": "lama_guanicoe", "prefix": "guanaco",
        "karyotype": "2n = 74",
        "groupes": {
            "Pays": {
                "ARGENTINE": ["ALG"],
                "USA":       ["CLG"],
            },
        },
    },
    "🦙 Lama (Lama glama)": {
        "taxon": "9844", "vep_species": "lama_glama", "prefix": "llama",
        "karyotype": "2n = 74",
        "groupes": {
            "Pays": {
                "ARGENTINE": ["ALL"],
                "BOLIVIE":   ["BLL"],
            },
        },
    },
    "🦙 Vigogne (Vicugna vicugna)": {
        "taxon": "9839", "vep_species": "vicugna_vicugna", "prefix": "vicuna",
        "karyotype": "2n = 76",
        "groupes": {
            "Sous-espèce": {
                "VICUGNA":          ["BVV", "UVV"],
                "MENSALIS":         ["BVM"],
                "SP_NON_DETERMINE": ["AVS", "BVS"],
            },
            "Pays": {
                "ARGENTINE": ["AVS"],
                "BOLIVIE":   ["BVM", "BVS", "BVV"],
                "USA":       ["UVV"],
            },
        },
    },
}

# Remplacer par votre utilisateur GitHub une fois le .map généré
GITHUB_RAW = "https://raw.githubusercontent.com/VOTRE_USER/nw-camelids-pipeline/main/data/reference"

MAP_SOURCES = {
    "Upload manuel (.map)":         None,
    "Générer depuis ancien .map":   None,
}

def get_config(species_key: str) -> dict:
    base = SPECIES_CONFIG[species_key]
    return {**base, "autosomes": AUTOSOMES, "sex_chr": SEX_CHR, "mt": MT,
            "ars": ARS_VICPAC4, "reference": REFERENCE}
