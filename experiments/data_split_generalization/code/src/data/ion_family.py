from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ").strip())


def normalize_key(value: Any) -> str:
    return normalize_text(value).lower()


def _match_any(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if pattern in text:
            return pattern
    return None


def classify_cation_family(full_name: Any, short_name: Any = "") -> tuple[str, str]:
    text = f"{normalize_key(full_name)} {normalize_key(short_name)}"
    if not text.strip():
        return "unknown_cation", "empty"

    rules = [
        ("benzimidazolium", ["benzimidazolium"]),
        ("imidazolium", ["imidazolium", "imidazol-3-ium", "imidazol-1-ium", "imidazolidinium", "imizodalium"]),
        ("pyridinium", ["pyridinium", "pydridinium", "pyridin-1-ium"]),
        ("pyrrolidinium", ["pyrrolidinium"]),
        ("piperidinium", ["piperidinium"]),
        ("morpholinium", ["morpholinium"]),
        ("ammonium", ["ammonium", "aminium", "azanium", "guanidinium", "diamine"]),
        ("phosphonium", ["phosphonium"]),
        ("sulfonium", ["sulfonium"]),
        ("cholinium", ["cholinium", "choline"]),
    ]
    for family, patterns in rules:
        matched = _match_any(text, patterns)
        if matched:
            return family, matched
    return "other_cation", "fallback"


def classify_anion_family(full_name: Any, short_name: Any = "") -> tuple[str, str]:
    text = f"{normalize_key(full_name)} {normalize_key(short_name)}"
    if not text.strip():
        return "unknown_anion", "empty"

    rules = [
        ("tfsamide", ["bis[(trifluoromethyl)sulfonyl]imide", "bis((trifluoromethyl)sulfonyl)amide", "bis(trifluoromethanesulfonyl)amide", "bis{(trifluomethyl)sulfonyl}imide", "trifluoro-n-[(trifluoromethyl)sulfonyl]methanesulfonamide", "sulfonyl]imide", "sulfonyl)imide", "sulfonyl}imide", "sulfonyl)amide", "sulfonamide", "sulfonimidate", "sulfonyl]methide", "brsi", "bpsi", "btsi", "bpla"]),
        ("fsamide", ["bis(fluorosulfonyl)amide", "bfsa"]),
        ("fluorophosphate", ["hexafluorophosphate", "tris(pentafluoroethyl)trifluorophosphate", "trifluorotris", "phosphate(v)", "hxpe", "tpps"]),
        ("fluoroborate", ["tetrafluoroborate", "trifluoroborate", "tetracyanoborate", "bis(oxalato)borate", "borate", "tefb", "tcbe", "bobe"]),
        ("halide", ["chloride", "bromide", "iodide", "fluoride", "chde", "brde", "iode"]),
        ("carboxylate", ["acetate", "formate", "propionate", "propanoate", "butanoate", "butyrate", "pentanoate", "hexanoate", "octanoate", "decanoate", "oleate", "lactate", "hydroxypropanoate", "glycolate", "glycollate", "benzoate", "salicylate", "trifluoroacetate", "carboxylate", "phenolate", "acetylacetonate", "acte", "fomt", "tfat"]),
        ("amino_acid", ["alaninate", "glycinate", "serinate", "prolinate", "valinate", "leucinate", "threoninate", "lysinate", "cysteinate", "phenylalaninate", "aspartate", "glutamate", "histidinate", "argininate", "methioninate", "tryptophanate", "tyrosinate", "asparaginate", "taurate", "taurinate", "glycine", "threonine"]),
        ("sulfate_sulfonate", ["sulfate", "sulfonate", "sulfonatooxy", "hydrogensulfate", "methanesulfonate", "trifluoromethanesulfonate", "tosylate", "docusate", "else", "mest", "hnse", "mefe", "tmsn"]),
        ("cyanamide_cyanocarbon", ["dicyanamide", "tricyanomethanide", "tricyanomethane", "thiocyanate", "cyanocyanamide", "dica", "tyae", "toce"]),
        ("nitrate", ["nitrate", "nite"]),
        ("carbonate", ["carbonate"]),
        ("halometalate", ["chloroaluminate", "dialuminate"]),
        ("perchlorate", ["perchlorate"]),
        ("phosphate_phosphinate", ["phosphate", "phosphonate", "phosphinate"]),
        ("heterocyclic", ["pyrazolide", "imidazolide", "triazolide", "tetrazolide", "indazolide", "pyrrolide", "saccharinate", "oxathiazin"]),
    ]
    for family, patterns in rules:
        matched = _match_any(text, patterns)
        if matched:
            return family, matched
    return "other_anion", "fallback"


def add_ion_family_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cation = [
        classify_cation_family(row.get("Cation_FullName"), row.get("Cation_ShortName"))
        for _, row in out.iterrows()
    ]
    anion = [
        classify_anion_family(row.get("Anion_FullName"), row.get("Anion_ShortName"))
        for _, row in out.iterrows()
    ]
    out["Cation_Family"] = [item[0] for item in cation]
    out["Cation_Family_Match"] = [item[1] for item in cation]
    out["Anion_Family"] = [item[0] for item in anion]
    out["Anion_Family_Match"] = [item[1] for item in anion]
    return out


def export_ion_family_report(df: pd.DataFrame, output_path: str | Path) -> Path:
    tagged = add_ion_family_columns(df)
    cation_cols = ["Cation_FullName", "Cation_ShortName", "Cation_Family", "Cation_Family_Match"]
    anion_cols = ["Anion_FullName", "Anion_ShortName", "Anion_Family", "Anion_Family_Match"]
    cations = (
        tagged[cation_cols]
        .drop_duplicates()
        .assign(ion_type="cation")
        .rename(
            columns={
                "Cation_FullName": "FullName",
                "Cation_ShortName": "ShortName",
                "Cation_Family": "Family",
                "Cation_Family_Match": "Match",
            }
        )
    )
    anions = (
        tagged[anion_cols]
        .drop_duplicates()
        .assign(ion_type="anion")
        .rename(
            columns={
                "Anion_FullName": "FullName",
                "Anion_ShortName": "ShortName",
                "Anion_Family": "Family",
                "Anion_Family_Match": "Match",
            }
        )
    )
    report = pd.concat([cations, anions], ignore_index=True)
    row_counts = []
    for ion_type, name_col, short_col, family_col in [
        ("cation", "Cation_FullName", "Cation_ShortName", "Cation_Family"),
        ("anion", "Anion_FullName", "Anion_ShortName", "Anion_Family"),
    ]:
        counts = (
            tagged.groupby([name_col, short_col, family_col], dropna=False)
            .size()
            .reset_index(name="row_count")
            .rename(columns={name_col: "FullName", short_col: "ShortName", family_col: "Family"})
        )
        counts["ion_type"] = ion_type
        row_counts.append(counts)
    counts_df = pd.concat(row_counts, ignore_index=True)
    report = report.merge(counts_df, on=["ion_type", "FullName", "ShortName", "Family"], how="left")
    report = report[["ion_type", "Family", "FullName", "ShortName", "Match", "row_count"]].sort_values(
        ["ion_type", "Family", "row_count", "FullName"],
        ascending=[True, True, False, True],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path
