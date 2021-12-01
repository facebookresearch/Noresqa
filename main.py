#Copyright (c) Meta Platforms, Inc. and affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.


import torch
import argparse
import librosa as librosa
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from model import NORESQA
from scipy import signal

def argument_parser():
    """
    Get an argument parser.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--GPU_id', help='GPU Id to use (-1 for cpu)', default=-1, type=int)
    parser.add_argument('--mode', choices=['file', 'list'], help='predict noresqa for test file with another file (mode = file) as NMR or, with a database given as list of files (mode=list) as NMRs', default='file', type=str)
    parser.add_argument('--test_file', help='test speech file', required=True, type=str, default='sample_clips/noisy.wav')
    parser.add_argument('--nmr', help='for mode=file, path of nmr speech file. for mode=list, path of text file which contains list of nmr paths', required = True, type=str, default='sample_clips/clean.wav')
    return parser

args = argument_parser().parse_args()


# Loading the model
model = NORESQA(output=40, output2=40)

# Loading checkpoint
model_checkpoint_path = 'models/model.pth'
state = torch.load(model_checkpoint_path,map_location="cpu")['state_base']

pretrained_dict = {}
for k, v in state.items():
    if 'module' in k:
        pretrained_dict[k.replace('module.','')]=v
    else:
        pretrained_dict[k]=v
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(pretrained_dict)

# device
if args.GPU_id >=0 and torch.cuda.is_available():
    device = torch.device("cuda:{}".format(args.GPU_id))
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

sfmax = nn.Softmax(dim=1)

# function extraction stft
def extract_stft(audio, sampling_rate = 16000):

    f, t, Sxx = signal.stft(audio, sampling_rate, window='hann',nperseg=512,noverlap=256,nfft=512)
    Sxx = Sxx[:256,:]

    feat = np.concatenate((np.abs(Sxx).reshape([Sxx.shape[0],Sxx.shape[1],1]), np.angle(Sxx).reshape([Sxx.shape[0],Sxx.shape[1],1])), axis=2)

    return feat 

# function doing prediction
def model_prediction(test_feat, nmr_feat):

    intervals_sdr = np.arange(0.5,40,1) 

    with torch.no_grad():
        
        ranking_frame,sdr_frame,snr_frame = model(test_feat.permute(0,3,2,1),nmr_feat.permute(0,3,2,1))
        # preference task prediction
        
        ranking = sfmax(ranking_frame).mean(2).detach().cpu().numpy()
        pout = ranking[0][0]

        # quantification task
        sdr = intervals_sdr * (sfmax(sdr_frame).mean(2).detach().cpu().numpy())
        qout = sdr.sum()

    return pout, qout

# function checking if the size of the inputs are same. If not, then the reference audio's size is adjusted
def check_size(audio_ref,audio_test):
    
    if len(audio_ref) > len(audio_test):
        print('Durations dont match. Adjusting duration of reference.')
        while len(audio_ref)>len(audio_test):
            audio_test = np.append(audio_test, audio_test)
        audio_test = audio_test[:len(audio_ref)]
        
    elif len(audio_ref) < len(audio_test):
        print('Durations dont match. Adjusting duration of reference.')
        audio_test = audio_test[:len(audio_ref)]  
    
    return audio_ref, audio_test

# reading audio clips
def audio_loading(path,sampling_rate=16000):

    audio, fs = librosa.load(path)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    if fs != sampling_rate:
        audio = librosa.resample(audio,fs,sampling_rate)

    return audio

# top level function combining loading, and feature extraction
def feats_loading(ref_path,test_path):
    
    audio_ref = audio_loading(ref_path)
    audio_test = audio_loading(test_path)

    audio_ref, audio_test = check_size(audio_ref,audio_test)

    ref_feat = extract_stft(audio_ref)
    test_feat = extract_stft(audio_test)

    return ref_feat,test_feat



if args.mode == 'file':

    nmr_feat,test_feat = feats_loading(args.nmr,args.test_file)
    test_feat = torch.from_numpy(test_feat).float().to(device).unsqueeze(0)
    nmr_feat = torch.from_numpy(nmr_feat).float().to(device).unsqueeze(0)

    pout, qout = model_prediction(test_feat,nmr_feat)
    
    print('Probaility of the test speech cleaner than the given NMR =', pout)
    print('NORESQA score of the test speech with respect to the given NMR =', qout)

    
elif args.mode == 'list':

    noresqa = []
    probs = []
    with open(args.nmr) as f:
        for ln in f:

            nmr_feat, test_feat = feats_loading(ln.strip(),args.test_file)
            nmr_feat = torch.from_numpy(nmr_feat).float().to(device).unsqueeze(0)
            test_feat = torch.from_numpy(test_feat).float().to(device).unsqueeze(0)
            pout, qout = model_prediction(test_feat,nmr_feat)
            noresqa.append(qout)
            probs.append(pout)
            print(f"Prob. of test cleaner than {ln.strip()} = {pout}. Noresqa score = {qout}")
    
    
