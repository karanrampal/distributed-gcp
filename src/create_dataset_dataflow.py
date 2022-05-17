#!/usr/bin/env python3
"""Apache beam pipeline for dataset creation"""

import argparse
import os
from typing import List, Tuple

from apache_beam import dataframe, Pipeline
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
        default=["fitted",
                 "regularfit",
                 "slimfit",
                 "oversized",
                 "skinnyfit",
                 "relaxedfit",
                 "loosefit",
                 "superskinnyfit",
                 "musclefit"],
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
#        padma.castor = padma.castor.astype(int)

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
        data.to_parquet(os.path.join(known_args.out_dir, "data.parquet"), index=False)

        # Merge castor data to get output
        out = castors.merge(
            data.set_index("castor"), left_on="castor", right_index=True, how="inner"
        )
        # Convert string labels to ints
#        out["labels"] = out["product_fit"].map(dict(zip(known_args.labels,
#                                                    range(len(known_args.labels)))))

        # ToDo Split data into training and test dataset
        # cval = StratifiedGroupKFold(n_splits=2)

        # Write output files
        out.to_csv(os.path.join(known_args.out_dir, "full_fit.csv"), index=False)

        # ToDo write train test split
        # train_fit.to_csv(os.path.join(known_args.out_dir, "train.csv"), index=False)
        # test_fit.to_csv(os.path.join(known_args.out_dir, "test.csv"), index=False)


if __name__ == "__main__":
    main()
