import torch
from joblib import Parallel, delayed
from algo.transfer import set_gin, get_checkpoint, get_audio, out_audio, fasterize, transfer
from neural_waveshaping_synthesis.data.utils.preprocess_audio import resample_audio
import numpy as np
from tqdm import tqdm

def get_checkpoints(instrs=['vn', 'fl', 'tpt']):
    checks = []
    for instr in instrs:
        net, data_mean, data_std = get_checkpoint(instr)
        net = fasterize(net)
        checks.append([net, data_mean, data_std])
        print(f'load {instr} net ...')
    return checks

device_str = 'cpu'
device     = torch.device(device_str)
set_gin(device_str)
checkpoints = get_checkpoints()


def algo_single(audio, net, data_mean, data_std):
    out = transfer(audio,
                   net,
                   data_mean,
                   data_std,
                   device)
    return out


def algo(audio_path, out_path, instr_index):
    net, data_mean, data_std = checkpoints[instr_index]
    print('get audio ...')
    audio, sr = get_audio(audio_path)
    audio = resample_audio(audio, sr, net.sample_rate)
    instr = ['vn', 'fl', 'tpt'][instr_index]
    print(f'transfer to {instr} ...')
    # audios = np.reshape(audio, (-1, 32000))
    
    # for audio in tqdm(audios):
    #     print(audio.shape)
    #     algo_single(audio, net, data_mean, data_std)
    # outs = Parallel(n_jobs=None)(delayed(algo_single)(audio, net, data_mean, data_std) for audio in tqdm(audios))
    # import pdb;pdb.set_trace()

    out = transfer(audio[:10*16000],
                   net,
                   data_mean,
                   data_std,
                   device)
    print('out audio ...', out_path)
    out_audio(out_path, out, sr)
