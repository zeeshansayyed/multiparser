from os import sep
from pathlib import Path
import pandas as pd

data_dir = Path('data/stdseg/')
exp_root = Path('exp/')
exp_type = 'stdseg'
exp_name = '1'
exp_dir = exp_root / exp_type / exp_name

gold_frame = pd.read_csv(
    data_dir / 'test.conll', sep='\t', header=None, dtype=object,
    names=['ID', 'RAW', 'SEG', 'SEGL', 'STD', 'STDSEG', 'STDSEGL'])

pred_frame = pd.read_csv(
    exp_dir / 'raw-std-test.conll', header=None, sep='\t', dtype=object,
    names=['ID', 'RAW', 'SEG', 'SEGL', 'STD', 'STDSEG', 'STDSEGL'])
