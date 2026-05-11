"""
One-shot helper: clone (or pull) the PGLib-UC repo into a sibling directory.

Default target: ./uc_experiment_v3/pglib-uc/
Override with --dest.
"""

import argparse
import os
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(THIS_DIR, 'pglib-uc')
REPO_URL = 'https://github.com/power-grid-lib/pglib-uc.git'


def fetch(dest=DEFAULT_DEST):
    if os.path.isdir(os.path.join(dest, '.git')):
        print(f"Updating existing clone at {dest}")
        return subprocess.call(['git', '-C', dest, 'pull', '--ff-only'])
    print(f"Cloning {REPO_URL} -> {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    return subprocess.call(['git', 'clone', '--depth', '1', REPO_URL, dest])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest', default=DEFAULT_DEST)
    args = parser.parse_args()
    rc = fetch(args.dest)
    if rc != 0:
        print(f"git failed with rc={rc}; you may need to clone manually:")
        print(f"  git clone {REPO_URL} {args.dest}")
        sys.exit(rc)
    print("Done.")


if __name__ == '__main__':
    main()
