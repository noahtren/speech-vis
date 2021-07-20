import torch
import fairseq

cp_path = './wav2vec_small.pt'
model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
model = model[0]
model.eval()


def embed_signal(wav_input_16khz):
  z = model.feature_extractor(wav_input_16khz)
  z = torch.swapaxes(z, 1, 2)
  qz, idxs = model.quantizer.forward_idx(z)
  return {'z': z, 'qz': qz, 'idxs': idxs}
