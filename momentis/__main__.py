import argparse
from momentis.momentis import main
from pathlib import Path
from .utils import parse_keywords


def parse_args() -> int:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH", help="Path to videos", type=str)
    parser.add_argument(
        "--output", help="Output folder. Default is {PATH}/opencv_output", default=None
    )
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug mode", default=False
    )

    args = parser.parse_args()
    try:
        input_path = Path(args.PATH)
        output_path = Path(args.output) if args.output is not None else input_path / "opencv_output"
    except AttributeError:
        print("Invalid arguments.")
        parser.print_help()
        return -1

    keywords = parse_keywords()
    main(input_path=input_path, keywords=keywords, output_path=output_path, debug=args.debug)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(parse_args())
