import warnings
warnings.filterwarnings("ignore")

import os
import gin
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.io import wavfile
from neural_waveshaping_synthesis.data.utils.loudness_extraction import extract_perceptual_loudness
from neural_waveshaping_synthesis.data.utils.f0_extraction       import extract_f0_with_crepe
from neural_waveshaping_synthesis.data.utils.preprocess_audio    import convert_to_float32_audio, make_monophonic, resample_audio
from neural_waveshaping_synthesis.models.modules.shaping         import FastNEWT
from neural_waveshaping_synthesis.models.neural_waveshaping      import NeuralWaveshaping


# Default Settings

gin_dir        = 'algo/gin'
checkpoint_dir = 'algo/checkpoints/nws'


def set_gin(device_str):
    gin.constant('device', device_str)
    gin.parse_config_file(os.path.join(gin_dir, 'models/newt.gin'))
    gin.parse_config_file(os.path.join(gin_dir, 'data/urmp_4second_crepe.gin'))
    
def get_audio(audio_path):
    audio, sr = sf.read(audio_path)
    # sr, audio = wavfile.read(audio_path)
    # audio     = convert_to_float32_audio(make_monophonic(audio))
    audio = make_monophonic(audio).astype(np.float32)

    return audio, sr

def out_audio(out_path, audio, sr):
    sf.write(out_path, audio, sr)


def get_checkpoint(instr): # vn, fl, tpt
    selected_path = os.path.join(checkpoint_dir, instr)
    net           = NeuralWaveshaping.load_from_checkpoint(os.path.join(selected_path, 'last.ckpt'))
    net.eval()
    data_mean = np.load(os.path.join(selected_path, 'data_mean.npy'))
    data_std  = np.load(os.path.join(selected_path, 'data_std.npy'))
    return net, data_mean, data_std


                      
def extract_feature(audio, 
                    data_mean, 
                    data_std , 
                    device   , 
                    pitch_shift        = 1, 
                    loudness_scale     = 0.5, 
                    pitch_smoothing    = 0,
                    loudness_smoothing = 0):
    
    with torch.no_grad():
        
        f0, confidence   = extract_f0_with_crepe(audio,
                                                 full_model=True,
                                                 maximum_frequency=1000)
        loudness         = extract_perceptual_loudness(audio)
        f0_shifted       = f0 * (2 ** pitch_shift)
        loudness_scaled  = loudness * loudness_scale
        loud_nrom        = (loudness_scaled - data_mean[1]) / data_std[1]

        f0_t             = torch.tensor(f0_shifted, device=device).float()
        loud_norm_t      = torch.tensor(loud_nrom , device=device).float()
        
        if pitch_smoothing != 0:
            f0_t = torch.nn.functional.conv1d(
                f0_t.expand(1, 1, -1),
                torch.ones(1, 1, pitch_smoothing * 2 + 1, device=device) /(pitch_smoothing * 2 + 1),
                padding=pitch_smoothing
            ).squeeze()
        f0_norm_t = torch.tensor((f0_t.cpu() - data_mean[0]) / data_std[0], device=device).float()
            
        if loudness_smoothing != 0:
            loud_norm_t = torch.nn.functional.conv1d(loud_norm_t.expand(1, 1, -1),
                                                     torch.ones(1, 
                                                                1, 
                                                                loudness_smoothing * 2 + 1, 
                                                                device=device) / (loudness_smoothing * 2 + 1),
                                                     padding=loudness_smoothing).squeeze()
        f0_norm_t = torch.tensor((f0_t.cpu() - data_mean[0]) / data_std[0], device=device).float()
        control = torch.stack((f0_norm_t, loud_norm_t), dim=0)

    return f0_t, control 

    
def fasterize(net, use_fastnewt=True):
    original_newt  = net.newt
    if use_fastnewt:
        net.newt = FastNEWT(original_newt)
    else:
        net.newt = original_newt
    return net


def transfer(audio, net, data_mean, data_std, device, params=None):
    net = fasterize(net)
    with torch.no_grad():
        if params is not None:
            f0_t, control = extract_feature(audio, data_mean, data_std, device, **params)
        else:
            f0_t, control = extract_feature(audio, data_mean, data_std, device)
        out = net(f0_t.expand(1, 1, -1), control.unsqueeze(0))
        out = out.detach().cpu().numpy()[0]
    return out


def main():
    instr      = 'vn' # set transfer instrument -> vn, fl, tpt
    audio_path = './audio/test.wav'
    out_path   = './audio/out.wav'
    device_str = 'cpu' # cpu, cuda
    device     = torch.device(device_str)
    
    params     = {'pitch_shift': 1,        # -4-4
                  'loudness_scale': 0.5,   #  0-1
                  'pitch_smoothing': 0,    #  0-100
                  'loudness_smoothing': 0} #  0-100

    set_gin(device_str)
    
    net, data_mean, data_std = get_checkpoint(instr)
    print('get checkpoints ...')
    net       = fasterize(net.to(device))
    print('fasterize ...')
    audio, sr = get_audio(audio_path)
    audio     = resample_audio(audio, sr, net.sample_rate)
    print('get audio ...')
    out       = transfer(audio, 
                         net, 
                         data_mean, 
                         data_std, 
                         device, 
                         params)
    print('transfer ...')
    out_audio(out_path, out, sr)
    print('out audio ...', out_path)


    
    
if __name__ == "__main__":
    main()
