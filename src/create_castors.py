#!/usr/bin/env python3
"""Script for creating castors metadata"""

import argparse
import os
from typing import List, Tuple

import gcsfs
import pandas as pd


def arg_parser() -> Tuple[argparse.Namespace, List[str]]:
    """Parse CLI argments"""
    parser = argparse.ArgumentParser(description="Create castors metadata")
    parser.add_argument(
        "-r",
        "--root",
        default="gs://hm-images-bucket/images/",
        type=str,
        help="Root directory",
    )
    parser.add_argument(
        "-p",
        "--proj_name",
        default="smle-attribution-d237",
        type=str,
        help="Name of project",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="gs://hm-images-bucket/annotations/",
        type=str,
        help="Output directory path",
    )

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = arg_parser()

    num = len(args.root)

    gfs = gcsfs.GCSFileSystem(project=args.proj_name)

    file_list = gfs.glob(args.root + "**/*.jpg", recursive=True)
    castors = [int(os.path.basename(path)[:-4]) for path in file_list]
    path_list = [path[num:] for path in file_list]

    out = pd.DataFrame(data={"path": path_list, "castor": castors})

    out.to_csv(os.path.join(args.outdir, "castors1.csv"), index=False)


if __name__ == "__main__":
    main()
