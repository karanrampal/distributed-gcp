#!/usr/bin/env python3
"""Apache beam pipeline for dataset creation"""

import argparse
import os
from typing import List, Tuple

from apache_beam import Pipeline
from apache_beam.dataframe.io import read_csv, read_parquet
from apache_beam.options.pipeline_options import PipelineOptions


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
        default="gs://hdl-tables/dim/dim_pim/*.parquet",
        type=str,
        help="Pim directory",
    )
    parser.add_argument(
        "-d",
        "--padma",
        default="gs://hdl-tables/dma/product_article_datamart/*.parquet",
        type=str,
        help="Padma directory",
    )
    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        default=[
            "fitted",
            "regularfit",
            "slimfit",
            "oversized",
            "skinnyfit",
            "relaxedfit",
            "loosefit",
            "superskinnyfit",
            "musclefit",
        ],
        help="List of labels",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        default="gs://hm-images-bucket/annotations/dataflow_output",
        type=str,
        help="Output directory",
    )

    return parser.parse_known_args()


def main() -> None:
    """Main function"""
    known_args, pipeline_args = arg_parser()

    with Pipeline(options=PipelineOptions(pipeline_args)) as abp:
        pim = abp | "ReadPimTable" >> read_parquet(
            known_args.pim, columns=["product_code", "article_code", "product_fit"]
        )
        padma = abp | "ReadPadmaTable" >> read_parquet(
            known_args.padma, columns=["product_code", "article_code", "castor"]
        )
        castors = abp | "ReadCastorFile" >> read_csv(
            os.path.join(known_args.annot, "castors.csv")
        )

        # Transform padma table
        padma = padma.drop_duplicates(keep="any")
        padma.castor = padma.castor.astype(int)

        # Transform pim table
        pim = pim.dropna(axis=0, subset=["article_code", "product_fit"])
        pim = pim.drop_duplicates(keep="any")

        # Merge pim and padma table
        data = pim.merge(
            padma.set_index(["product_code", "article_code"]),
            left_on=["product_code", "article_code"],
            right_index=True,
            how="left",
        )
        data = data.drop(axis=1, labels=["product_code", "article_code"])
        data = data[~data["product_fit"].str.contains("[", regex=False)]

        # Merge castor data to get output
        out = castors.merge(
            data.set_index("castor"), left_on="castor", right_index=True, how="inner"
        )
        # Convert string labels to ints
        out["labels"] = out["product_fit"].map(
            dict(zip(known_args.labels, range(len(known_args.labels))))
        )

        # Split data into training and test dataset
        tmp = out[["product_fit", "castor"]].drop_duplicates(keep="any")
        sub_train = tmp.groupby("product_fit").sample(frac=0.8)
        sub_train["is_train"] = True
        final = out.merge(
            sub_train[["castor", "is_train"]].set_index("castor"),
            left_on="castor",
            right_index=True,
            how="left",
        )
        final.fillna(False, inplace=True)
        train_fit = final.loc[
            final.is_train, ["path", "castor", "product_fit", "labels"]
        ]
        test_fit = final.loc[
            ~final.is_train, ["path", "castor", "product_fit", "labels"]
        ]

        # Write output files
        out.to_csv(os.path.join(known_args.out_dir, "full_fit.csv"), index=False)
        train_fit.to_csv(os.path.join(known_args.out_dir, "train.csv"), index=False)
        test_fit.to_csv(os.path.join(known_args.out_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
