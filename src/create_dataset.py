#!/usr/bin/env python3
"""Script for dataset creation"""

import argparse
import os
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def arg_parser() -> Tuple[argparse.Namespace, List[str]]:
    """Parse CLI argments"""
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "-c",
        "--castors",
        default="gs://hm-images-bucket/annotations/castors.csv",
        type=str,
        help="Castors file",
    )
    parser.add_argument(
        "-p",
        "--pim",
        default="gs://hdl-tables/dim/dim_pim/",
        type=str,
        help="Directory for Pim table",
    )
    parser.add_argument(
        "-d",
        "--padma",
        default="gs://hdl-tables/dma/product_article_datamart/",
        type=str,
        help="Directory for Padma table",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="gs://hm-images-bucket/annotations/",
        type=str,
        help="Output directory",
    )

    return parser.parse_known_args()


def main() -> None:
    """Main function"""
    known_args, _ = arg_parser()

    # Read input data
    padma = pd.read_parquet(
        known_args.padma, columns=["product_code", "article_code", "castor"]
    )
    pim = pd.read_parquet(
        known_args.pim, columns=["product_code", "article_code", "product_fit"]
    )
    castors = pd.read_csv(known_args.castors)

    # Clean data tables
    padma = padma.drop_duplicates()
    padma.castor = padma.castor.astype(int)
    pim = pim.dropna(axis=0, subset=["article_code", "product_fit"])
    pim = pim.drop_duplicates()

    # Merge pim and padma table
    data = pim.merge(padma, on=["product_code", "article_code"], how="left")
    data = data.drop(axis=1, labels=["product_code", "article_code"])
    data = data[~data["product_fit"].str.contains("[", regex=False)]

    # Merge castor data to get output
    out = castors.merge(data, on="castor", how="inner")
    out["labels"] = out["product_fit"].astype("category").cat.codes

    # Split data into training and test dataset
    cval = StratifiedGroupKFold(n_splits=2)
    train_idxs, test_idxs = next(cval.split(out.path, out.labels, out.castor))
    train_fit = out.iloc[train_idxs, :]
    test_fit = out.iloc[test_idxs, :]

    # Create automl data
    out_gcp = out[["path", "product_fit"]].copy()
    out_gcp["path"] = "gs://hm-images-bucket/images/" + out_gcp["path"]
    out_gcp["mode"] = "VALIDATION"
    out_gcp.loc[train_idxs, "mode"] = "TRAINING"
    out_gcp = out_gcp[["mode", "path", "product_fit"]]

    # Write output files
    out.to_csv(os.path.join(known_args.out_dir, "full_fit1.csv"), index=False)
    train_fit.to_csv(os.path.join(known_args.out_dir, "train1.csv"), index=False)
    test_fit.to_csv(os.path.join(known_args.out_dir, "test1.csv"), index=False)
    out_gcp.to_csv(
        os.path.join(known_args.out_dir, "full_fit_gcai1.csv"),
        index=False,
        header=False,
    )


if __name__ == "__main__":
    main()
