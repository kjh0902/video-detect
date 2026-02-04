import os
import glob
import math
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch

from scipy.io import wavfile
from scipy import signal
import python_speech_features
from SyncNetModel import *

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

class SyncNetInstance(torch.nn.Module):
    def __init__(self, dropout=0, num_layers_in_fc_layers=1024, device="cpu"):
        super().__init__()
        self.device = torch.device("cpu")
        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).to(self.device)
        self.__S__.eval()

    def _prepare_tmp_dir(self, opt):
        work_dir = os.path.join(opt.tmp_dir, opt.reference)
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir, exist_ok=True)
        return work_dir

    def _ffmpeg_extract_frames(self, videofile, out_pattern):
        cmd = (
            f'ffmpeg -y -i "{videofile}" -threads 1 '
            f'-vf fps=25 -vsync 0 -q:v 2 -f image2 "{out_pattern}"'
        )
        subprocess.check_call(cmd, shell=True)

    def _ffmpeg_extract_audio(self, videofile, out_wav):
        cmd = (
            f'ffmpeg -y -i "{videofile}" -vn -ac 1 -ar 16000 -acodec pcm_s16le "{out_wav}"'
        )
        subprocess.check_call(cmd, shell=True)

    def _load_frames_as_tensor(self, frame_files):
        images = []
        for fname in frame_files:
            img = cv2.imread(fname)
            if img is None:
                continue
            images.append(img)

        if len(images) < 5:
            return None, 0  

        im = np.stack(images, axis=0)  
        im = np.transpose(im, (3, 0, 1, 2))  
        im = np.expand_dims(im, axis=0)  
        imtv = torch.from_numpy(im.astype(np.float32)) 

        return imtv, len(images)

    def _load_audio_as_tensor(self, audio_wav_path, num_frames, fps=25, sr=16000):
        sample_rate, audio = wavfile.read(audio_wav_path)

        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.int16, copy=False)

        target_sec = float(num_frames) / float(fps)
        target_samples = int(round(target_sec * sr))

        if len(audio) < target_samples:
            pad = np.zeros((target_samples - len(audio),), dtype=np.int16)
            audio = np.concatenate([audio, pad], axis=0)
        elif len(audio) > target_samples:
            audio = audio[:target_samples]

        mfcc = python_speech_features.mfcc(audio, sr)
        mfcc = np.stack(list(zip(*mfcc)), axis=0)  

        cc = np.expand_dims(np.expand_dims(mfcc.astype(np.float32), axis=0), axis=0)
        cct = torch.from_numpy(cc)

        return audio, cct

    def evaluate(self, opt, videofile):
        self.__S__.eval()

        work_dir = self._prepare_tmp_dir(opt)
        frame_pattern = os.path.join(work_dir, "%06d.jpg")
        audio_wav = os.path.join(work_dir, "audio_raw.wav")

        self._ffmpeg_extract_frames(videofile, frame_pattern)
        self._ffmpeg_extract_audio(videofile, audio_wav)

        flist = sorted(glob.glob(os.path.join(work_dir, "*.jpg")))
        imtv, num_frames = self._load_frames_as_tensor(flist)
        if imtv is None:
            raise RuntimeError("Not enough frames extracted for SyncNet (need at least 5 frames).")

        audio, cct = self._load_audio_as_tensor(audio_wav, num_frames, fps=25, sr=16000)

        vid_sec = float(num_frames) / 25.0
        aud_sec = float(len(audio)) / 16000.0
        if abs(aud_sec - vid_sec) > 1e-3:
            print(f"WARNING: Audio ({aud_sec:.4f}s) and video ({vid_sec:.4f}s) lengths differ.")

        min_length = min(num_frames, math.floor(len(audio) / 640))
        lastframe = min_length - 5
        if lastframe <= 0:
            raise RuntimeError("Clip too short after alignment. Need longer video/audio.")

        im_feat = []
        cc_feat = []

        tS = time.time()
        with torch.no_grad():
            for i in range(0, lastframe, opt.batch_size):
                end = min(lastframe, i + opt.batch_size)

                im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, end)]
                im_in = torch.cat(im_batch, dim=0).to(self.device)
                im_out = self.__S__.forward_lip(im_in)
                im_feat.append(im_out.cpu())

                cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in range(i, end)]
                cc_in = torch.cat(cc_batch, dim=0).to(self.device)
                cc_out = self.__S__.forward_aud(cc_in)
                cc_feat.append(cc_out.cpu())

        im_feat = torch.cat(im_feat, dim=0)
        cc_feat = torch.cat(cc_feat, dim=0)

        print("Compute time %.3f sec." % (time.time() - tS))

        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists, dim=1), dim=1)

        minval, minidx = torch.min(mdist, dim=0)

        offset = opt.vshift - int(minidx.item())
        conf = float((torch.median(mdist) - minval).item())

        fdist = np.stack([dist[minidx].numpy() for dist in dists])
        fconf = float(torch.median(mdist).item()) - fdist
        fconfm = signal.medfilt(fconf, kernel_size=9)

        np.set_printoptions(formatter={"float": "{: 0.3f}".format})
        print("Framewise conf: ")
        print(fconfm)
        print("AV offset:\t%d\nMin dist:\t%.3f\nConfidence:\t%.3f" % (offset, float(minval.item()), conf))

        dists_npy = np.array([dist.numpy() for dist in dists])
        return offset, conf, dists_npy

    def extract_feature(self, opt, videofile):
        self.__S__.eval()

        cap = cv2.VideoCapture(videofile)
        images = []
        while True:
            ret, image = cap.read()
            if not ret:
                break
            images.append(image)
        cap.release()

        if len(images) < 5:
            raise RuntimeError("Not enough frames to extract lip features (need at least 5 frames).")

        im = np.stack(images, axis=0)  
        im = np.transpose(im, (3, 0, 1, 2))  
        im = np.expand_dims(im, axis=0)  
        imtv = torch.from_numpy(im.astype(np.float32))

        lastframe = len(images) - 4
        im_feat = []

        tS = time.time()
        with torch.no_grad():
            for i in range(0, lastframe, opt.batch_size):
                end = min(lastframe, i + opt.batch_size)
                im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, end)]
                im_in = torch.cat(im_batch, dim=0).to(self.device)
                im_out = self.__S__.forward_lipfeat(im_in)
                im_feat.append(im_out.cpu())

        im_feat = torch.cat(im_feat, dim=0)
        print("Compute time %.3f sec." % (time.time() - tS))
        return im_feat

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location="cpu")

        self_state = self.__S__.state_dict()
        for name, param in loaded_state.items():
            if name in self_state:
                self_state[name].copy_(param)