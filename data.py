import os

import numpy as np
import pandas as pd
import librosa

commonvoice_base = os.path.expanduser('~/Code/s2021/audio/data/en/')
df = pd.read_csv(os.path.join(commonvoice_base, 'train.tsv'), sep='\t')


def get_example(idx):
  signal, _ = librosa.load(os.path.join(commonvoice_base, 'clips',
                                        df['path'].iloc[idx] + '.mp3'),
                           sr=16_000)
  return {'signal': signal, 'sentence': df['sentence'].iloc[idx]}
