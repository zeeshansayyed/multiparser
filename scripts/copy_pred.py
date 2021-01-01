from pathlib import Path
import shutil

name = 'padt-tree-proj'
source_root = Path("exp/arabic/ud-sud") / name
dest_root = Path("/N/slate/zasayyed/Projects/results") / name


for source_dir in source_root.glob('*'):
    print(source_dir.name)
    dest_dir = dest_root / source_dir.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for pred_file in source_dir.glob('*.conllu'):
        print(pred_file)
        shutil.copy(pred_file, dest_dir)
    print()