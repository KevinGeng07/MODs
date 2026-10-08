"""python -m mods demo [--port 8000] [--data-dir .mods]: serve the MNIST demo."""

import argparse


def main():
    p = argparse.ArgumentParser(prog="python -m mods")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="serve the MNIST demo dashboard")
    demo.add_argument("--port", type=int, default=8000)
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--data-dir", default=".mods", help="where modsdb keeps its log and snapshots")
    a = p.parse_args()

    from .demo import MnistCNN
    MnistCNN.serve(port=a.port, host=a.host, data_dir=a.data_dir)


if __name__ == "__main__":  # required: DataLoader workers are spawned and re-import this module
    main()
