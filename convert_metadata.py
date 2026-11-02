"""
convert_metadata.py — Convertit l'Excel de métadonnées en CSV
Usage : python convert_metadata.py NWC_Meta_Sample_Details_Suheil.xlsx
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd


def short_iid(sample_filename: str) -> str:
    name = str(sample_filename).replace(".CEL", "")
    parts = name.split("_")
    for p in parts:
        if len(p) >= 5 and p[:3] in ("ARG", "BOL", "PER", "USA") and p[3:].isdigit():
            return p
    return name.split("_")[2] if len(parts) > 2 else name


def main(xlsx_path: str) -> None:
    xlsx = Path(xlsx_path)
    if not xlsx.exists():
        sys.exit(f"❌ Fichier introuvable : {xlsx}")

    df_qc    = pd.read_excel(xlsx, sheet_name="Tabelle1")
    df_meta  = pd.read_excel(xlsx, sheet_name="Sample_Details")

    meta = df_meta.iloc[:, 8:].copy()
    meta.columns = [
        "Sample_Filename", "Country_Code", "Breed_Code", "blank1",
        "Country_Name", "Country_Code_2", "Breed_Code_2",
        "Species", "Scientific_Name", "Breed",
        "N", "N_Pass", "N_Fail",
    ]
    meta = meta.dropna(subset=["Sample_Filename"])
    meta = meta[meta["Sample_Filename"].astype(str).str.contains(".CEL", na=False)]

    qc = df_qc[["Sample Filename", "Pass/Fail", "DQC", "call_rate", "het_rate"]].copy()
    qc.columns = ["Sample_Filename", "Pass_Fail", "DQC", "call_rate", "het_rate"]

    df = meta.merge(qc, on="Sample_Filename", how="left")
    df_pass = df[df["Pass_Fail"] == "Pass"].copy()
    
    print(f"Échantillons total    : {len(df):>4}")
    print(f"Échantillons Pass     : {len(df_pass):>4}")
    print(f"Échantillons Fail     : {len(df) - len(df_pass):>4}")

    df_pass["IID"] = df_pass["Sample_Filename"].apply(short_iid)
    df_pass["FID"] = df_pass["Country_Code"].astype(str) + "_" + df_pass["Breed_Code"].astype(str)

    final = df_pass[[
        "FID", "IID", "Country_Code", "Country_Name",
        "Breed_Code", "Species", "Scientific_Name", "Breed",
        "DQC", "call_rate", "het_rate", "Pass_Fail", "Sample_Filename",
    ]].copy()

    print("\n📊 Résumé par espèce :")
    print(final.groupby("Species").size().to_string())
    
    out = xlsx.parent / "nwc_metadata.csv"
    final.to_csv(out, index=False)
    print(f"\n✅ Fichier écrit : {out}")
    print(f"   {len(final)} lignes × {len(final.columns)} colonnes")


if __name__ == "__main__":
    xlsx_path = sys.argv[1] if len(sys.argv) > 1 else "NWC_Meta_Sample_Details_Suheil.xlsx"
    main(xlsx_path)
