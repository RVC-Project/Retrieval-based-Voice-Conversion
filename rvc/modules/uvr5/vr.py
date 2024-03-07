import logging
import os

import librosa
import numpy as np
import soundfile as sf
import torch

from rvc.configs.config import Config
from rvc.lib.uvr5_pack.lib_v5 import nets_61968KB as Nets
from rvc.lib.uvr5_pack.lib_v5 import spec_utils
from rvc.lib.uvr5_pack.lib_v5.model_param_init import ModelParameters
from rvc.lib.uvr5_pack.lib_v5.nets_new import CascadedNet
from rvc.lib.uvr5_pack.utils import inference

logger = logging.getLogger(__name__)


class AudioPreprocess:
    def __init__(self, model_path, agg, tta=False):
        self.model_path = model_path
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": tta,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        self.config: Config = Config()
        self.version = 3 if "DeEcho" not in self.model_path else 2
        self.mp: ModelParameters = ModelParameters(
            f"rvc/lib/uvr5_pack/lib_v5/modelparams/4band_v{self.version}.json"
        )
        self.model = (
            Nets.CascadedASPPNet(self.mp.param["bins"] * 2)
            if self.version == 3
            else CascadedNet(
                self.mp.param["bins"] * 2, 64 if "DeReverb" in model_path else 48
            )
            .load_state_dict(torch.load(model_path, map_location="cpu"))
            .eval()
        )
        if self.config.is_half:
            self.model = self.model.half()
        self.model.to(self.config.device)

    def process(
        self,
        music_file,
    ):
        x_wave, y_wave, x_spec_s, y_spec_s = {}, {}, {}, {}
        bands_n = len(self.mp.param["band"])

        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                # librosa loading may be buggy for some audio. ffmpeg will solve this, but it's a pain
                x_wave[d] = librosa.core.load(
                    music_file,
                    sr=bp["sr"],
                    mono=False,
                    dtype=np.float32,
                    res_type=bp["res_type"],
                )[0]
                if x_wave[d].ndim == 1:
                    x_wave[d] = np.asfortranarray([x_wave[d], x_wave[d]])
            else:  # lower bands
                x_wave[d] = librosa.core.resample(
                    x_wave[d + 1],
                    orig_sr=self.mp.param["band"][d + 1]["sr"],
                    target_sr=bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            x_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                x_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )

            # pdb.set_trace()

        input_high_end_h = (
            self.mp.param["band"][1]["n_fft"] // 2
            - self.mp.param["band"][1]["crop_stop"]
        ) + (self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"])
        input_high_end = x_spec_s[1][
            :,
            self.mp.param["band"][1]["n_fft"] // 2
            - input_high_end_h : self.mp.param["band"][1]["n_fft"] // 2,
            :,
        ]
        x_spec_m = spec_utils.combine_spectrograms(x_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }
        with torch.no_grad():
            pred, x_mag, x_phase = inference(
                x_spec_m, self.config.device, self.model, aggressiveness, self.data
            )
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(x_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * x_phase
        v_spec_m = x_spec_m - y_spec_m

        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], y_spec_m, input_high_end, self.mp
            )
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                y_spec_m, self.mp, input_high_end_h, input_high_end_
            )
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], v_spec_m, input_high_end, self.mp
            )
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(
                v_spec_m, self.mp, input_high_end_h, input_high_end_
            )
        else:
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)

        return (
            (np.array(wav_instrument) * 32768).astype("int16"),
            (np.array(wav_vocals) * 32768).astype("int16"),
            self.mp.param["sr"],
            self.data["agg"],
        )
