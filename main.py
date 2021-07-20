import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import flax.linen as nn
import soundfile as sf
from google.cloud import speech

from data import get_example
from fairseq_wav import embed_signal

# use codebook
entries = json.load(open("codes.json", "r"))
client = speech.SpeechClient()


def render_cols(wav_out,
                signal,
                cols_per_seg=8,
                max_deviation=32,
                img_height=256,
                col_height_max=92,
                min_width=5):
  # setup
  cols = []
  idxs = wav_out['idxs'][0, :, 0].detach().numpy()
  sidxs = wav_out['idxs'][0, :, 1].detach().numpy()
  num_pixels_wide = cols_per_seg * len(idxs)
  reduce_size = signal.shape[0] // num_pixels_wide
  # pitch
  spec, f, _, img = plt.specgram(signal, NFFT=320, Fs=2)
  f = f * 16_000
  low_enough = np.where(f < 6000)
  pitch = (f[low_enough][:, np.newaxis] *
           nn.softmax(spec[low_enough] * 1024, axis=0)).sum(axis=0)
  pitch = (np.log(pitch) - np.log(100)) / (np.log(6000) - np.log(100))
  # mag
  mag = np.convolve(np.abs(signal), np.ones(
      reduce_size * 2)) / (reduce_size * 2)
  mag = mag[::reduce_size]
  mag = (mag / mag.max()) * 0.99 + 0.01
  avg_mag = np.convolve(mag, np.ones(12)) / 12

  for i, idx in enumerate(idxs[:-1]):
    cur_entry = {"1": entries["1"][str(idx)], "2": entries["2"][str(idx)]}
    next_entry = {
        "1": entries["1"][str(idxs[i + 1])],
        "2": entries["2"][str(idxs[i + 1])]
    }
    cur_color = {
        "1": np.array(cm.viridis(cur_entry["1"] / (np.pi * 2)))[:3],
        "2": np.array(cm.viridis(cur_entry["2"] / (np.pi * 2)))[:3]
    }
    next_color = {
        "1": np.array(cm.viridis(next_entry["1"] / (np.pi * 2)))[:3],
        "2": np.array(cm.viridis(next_entry["2"] / (np.pi * 2)))[:3]
    }
    cur_pos = (pitch[i] * 2) - 1
    next_pos = (pitch[i + 1] * 2) - 1
    for si in range(cols_per_seg):
      lerp_val = (cols_per_seg - si) / cols_per_seg
      color = {
          "1": cur_color["1"] * lerp_val + next_color["1"] * (1 - lerp_val),
          "2": cur_color["2"] * lerp_val + next_color["2"] * (1 - lerp_val),
      }
      pos = cur_pos * lerp_val + next_pos * (1 - lerp_val)
      center = img_height // 2 - int(pos * max_deviation)
      width = int(mag[len(cols)] * col_height_max)
      col = np.ones(shape=[img_height, 3])
      outer_color_idxs = np.arange(center - width, center + width)
      inner_color_idxs = np.arange(center - width // 2, center + width // 2)
      if avg_mag[len(cols)] * col_height_max < min_width:
        color = {
            k: v * (width / min_width) + np.array([1.0, 1, 1]) *
            (1.0 - (width / min_width))
            for k, v in color.items()
        }
      col[outer_color_idxs] = color["2"]
      col[inner_color_idxs] = color["1"]
      cols.append(col)
  img = np.stack(cols, axis=1)
  return img, pitch


def plot_example(example, ax):
  signal = example['signal']
  # get transcript
  sf.write("tmp.flac", signal, 16_000)
  os.system("gsutil cp ./tmp.flac gs://noahtren-research/tmp.flac")
  response = client.recognize(
      config=speech.RecognitionConfig(
          encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
          sample_rate_hertz=16000,
          language_code="en-US",
          enable_word_time_offsets=True),
      audio=speech.RecognitionAudio(uri="gs://noahtren-research/tmp.flac"))
  words = response.results[0].alternatives[0].words
  # get visualization
  wav_out = embed_signal(torch.Tensor(example['signal'][np.newaxis]))
  img, pitch = render_cols(wav_out, signal)
  ax.imshow(img)
  ax.axis('off')
  signal_seconds = signal.shape[0] / 16000
  for word in words:
    start_time = word.start_time.seconds + word.start_time.microseconds * 1e-6
    end_time = word.end_time.seconds + word.end_time.microseconds * 1e-6
    ax.text((((start_time + end_time) / 2) / signal_seconds) * img.shape[1], 0,
            word.word)


if __name__ == "__main__":
  fig, axes = plt.subplots(2, 2)
  for i in range(4):
    plot_example(get_example(i + 4), axes[i % 2][i // 2])

  plt.show()
