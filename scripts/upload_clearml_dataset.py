"""One-off helper: package a local ESP-cache directory as a versioned ClearML
Dataset and upload it, so the GPU rental box can `Dataset.get(...).get_local_copy()`
instead of us shipping tens of GB through -v mounts or scp.

Usage:
    python scripts/upload_clearml_dataset.py \\
        --project quannet-ssl --name sabdab_esp --path "C:/Users/admin/PycharmProjects/data/sabdab_esp_full/artefacts" \\
        --tags sabdab full_build
"""
from clearml import Dataset

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--path', required=True, help='local directory to package as the dataset root')
    ap.add_argument('--tags', nargs='*', default=None)
    args = ap.parse_args()

    ds = Dataset.create(dataset_name=args.name, dataset_project=args.project, dataset_tags=args.tags)
    print(f'Created dataset {ds.id} ({args.project}/{args.name}); adding files from {args.path} ...', flush=True)
    ds.add_files(path=args.path)
    print('Uploading ...', flush=True)
    ds.upload()
    ds.finalize()
    print(f'Done. dataset_id={ds.id}', flush=True)


if __name__ == '__main__':
    main()
