#!/usr/bin/env python3
"""Apache beam pipeline for dataset creation"""

import argparse
import os
from typing import List, Tuple

from apache_beam import Pipeline
from apache_beam.dataframe.io import read_csv, read_parquet
from apache_beam.options.pipeline_options import PipelineOptions
from sklearn.model_selection import StratifiedGroupKFold


def arg_parser() -> Tuple[argparse.Namespace, List[str]]:
    """Parse CLI argments"""
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument(
        "-a",
        "--annot",
        default="gs://hm-images-bucket/annotations",
        type=str,
        help="Root directory",
    )
    parser.add_argument(
        "-p",
        "--pim",
        default="gs://hdl-tables/dim/dim_pim",
        type=str,
        help="Pim directory",
    )
    parser.add_argument(
        "-d",
        "--padma",
        default="gs://hdl-tables/dma/product_article_datamart",
        type=str,
        help="Padma directory",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="gs://hm-images-bucket/annotations",
        type=str,
        help="Output directory",
    )

    return parser.parse_known_args()


def main() -> None:
    """Main function"""
    known_args, pipeline_args = arg_parser()

    with Pipeline(options=PipelineOptions(pipeline_args)) as abp:
        pim = abp | "ReadPimTable" >> read_parquet(known_args.pim)
        padma = abp | "ReadPadmaTable" >> read_parquet(known_args.padma)
        castors = abp | "ReadCastorFile" >> read_csv(
            os.path.join(known_args.annot, "castors.csv")
        )

        # Transform padma table
        padma = padma[["product_code", "article_code", "castor"]].drop_duplicates()
        padma.castor = padma.castor.astype(int)

        # Transform pim table
        pim = pim[["product_code", "article_code", "product_fit"]].drop_duplicates()
        pim = pim.dropna(axis=0, subset=["article_code", "product_fit"])

        # Merge pim and padma table
        data = pim.merge(padma, on=["product_code", "article_code"], how="left")
        data = data.drop(axis=1, labels=["product_code", "article_code"])
        data = data[~data["product_fit"].str.contains("[", regex=False)]

        # Create output data
        out = castors.merge(data, on="castor", how="inner")
        out["labels"] = out["product_fit"].astype("category").cat.codes

        # Split data into training and test dataset
        cval = StratifiedGroupKFold(n_splits=2)
        train_idxs, test_idxs = next(cval.split(out.path, out.labels, out.castor))
        train_fit = out.iloc[train_idxs, :]
        test_fit = out.iloc[test_idxs, :]

        # Write output files
        out.to_csv(os.path.join(known_args.out_dir, "full_fit.csv"), index=False)
        train_fit.to_csv(os.path.join(known_args.out_dir, "train.csv"), index=False)
        test_fit.to_csv(os.path.join(known_args.out_dir, "test.csv"), index=False)

        out_gcp = out[["path", "product_fit"]].copy()
        out_gcp["path"] = "gs://hm-images-bucket/images/" + out_gcp["path"]
        out_gcp["mode"] = "VALIDATION"
        out_gcp.loc[train_idxs, "mode"] = "TRAINING"
        out_gcp = out_gcp[["mode", "path", "product_fit"]]
        out_gcp.to_csv(
            os.path.join(known_args.out_dir, "full_fit_gcai.csv"),
            index=False,
            header=False,
        )


if __name__ == "__main__":
    main()
