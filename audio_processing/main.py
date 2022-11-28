import argparse
from audio_processing.audio_dataset_loader import (
    AudioDatasetLoader,
    ModelRefiner,
    connect_ssh,
)


def main(
    input_root_path: str,
    output_root_path: str,
    rewrite_data: bool,
    evaluation_percent: int,
):

    _ = ModelRefiner(
        input_path=input_root_path,
        output_path=output_root_path,
        rewrite_data=rewrite_data,
        evaluation_percent=evaluation_percent,
    )


def parse():
    parser = argparse.ArgumentParser(
        description="This script loads audio dataset from root path of database."
    )
    parser.add_argument(
        "-ir",
        "--input_root_path",
        help="Input root path for database",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-or",
        "--output_root_path",
        help="Output root path for database",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-rd", "--rewrite_data", help="Forcing rewriting data", action="store_true"
    )
    parser.add_argument(
        "-ep",
        "--evaluation_percent",
        help="Percentage of audio files used in GMM evaluation",
        required=True,
        type=float,
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    connect_ssh()

    main(
        input_root_path=args.input_root_path,
        output_root_path=args.output_root_path,
        rewrite_data=args.rewrite_data,
        evaluation_percent=args.evaluation_percent,
    )
