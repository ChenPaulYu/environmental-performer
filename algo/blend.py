import torch
import torchaudio.functional as F


def blend(audio1, audio2, ratio, sr):

    audio1 = F.lowpass_biquad(torch.tensor(audio1), sr, (sr/2)*ratio)
    audio2 = F.highpass_biquad(torch.tensor(audio2), sr, (sr/2)*ratio)

    if audio1.size(0) > audio2.size(0):
        audio1 = audio1[:audio2.size(0)]
    else:
        audio2 = audio2[:audio1.size(0)]

    return torch.lerp(audio2, audio1, ratio).numpy()