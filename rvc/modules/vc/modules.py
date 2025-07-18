import logging
import traceback
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from rvc.configs.config import Config
from rvc.lib.audio import load_audio, wav2
from rvc.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc.modules.vc.pipeline import Pipeline
from rvc.modules.vc.utils import *

logger: logging.Logger = logging.getLogger(__name__)


class VC:
    def __init__(self):
        self.n_spk: any = None
        self.tgt_sr: int | None = None
        self.net_g = None
        self.pipeline: Pipeline | None = None
        self.cpt: OrderedDict | None = None
        self.version: str | None = None
        self.if_f0: int | None = None
        self.version: str | None = None
        self.hubert_model: any = None

        self.config = Config()

    def get_vc(self, sid: str | Path, *to_return_protect: int):
        logger.info("Get sid: " + os.path.basename(sid) if hasattr(sid, "name") or isinstance(sid, str) else "ERROR")

        return_protect = [
            to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5,
            to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33,
        ]

        if hasattr(sid, "name"):
            sid = sid.name
        elif not isinstance(sid, str):
            raise RuntimeError(f"pathlib.Path or str expected for sid. Got {type(sid)}")

        weight_root = os.getenv("weight_root")
        person = sid if os.path.exists(sid) else f'{weight_root if weight_root is not None else "."}/{sid}'
        logger.info(f"Loading: {person}")
        
        if not os.path.exists(person) or person.endswith("/"):
            raise FileNotFoundError(f"model file not found (path: {person}).")
        
        self.cpt = torch.load(person, weights_only=False, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        if len(sid) == 0:
            logger.info("Clean model cache")
            del (self.hubert_model, self.tgt_sr, self.net_g)
            (self.net_g) = self.n_spk = index = None

        else:
            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.config.device)
            self.net_g = (
                self.net_g.half() if self.config.is_half else self.net_g.float()
            )

            self.pipeline = Pipeline(self.tgt_sr, self.config)
            self.n_spk = self.cpt["config"][-3]
            index = get_index_path_from_model(sid)
            logger.info("Select index: " + index)

        return self.n_spk, return_protect, index

    def vc_inference(
        self,
        sid: int,
        input_audio_path: Path | str,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        f0_file: Path | str | None = None,
        index_file: Path | str | None = None,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        hubert_path: str | Path | None = None,
    ):
        if hubert_path is None:
            hubert_path = os.getenv("hubert_path")
        elif hasattr(hubert_path, "name"):
            hubert_path = hubert_path.name
        elif not isinstance(hubert_path, str):
            raise RuntimeError(f"pathlib.Path, str, or None expected for hubert_path. Got {type(hubert_path)}")
        
        if hubert_path is None or not os.path.exists(hubert_path):
            raise FileNotFoundError("hubert_path not found.")

        if hasattr(input_audio_path, "name"):
            input_audio_path = input_audio_path.name
        elif not isinstance(input_audio_path, str):
            raise RuntimeError(f"pathlib.Path or str expected for input_audio_path. Got {type(input_audio_path)}")
        
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError("input_audio_path not found.")
        
        if isinstance(f0_file, str):
            f0_file = Path(f0_file)
        elif not isinstance(f0_file, Path) and f0_file is not None:
            raise RuntimeError(f"pathlib.Path, str, or None expected for f0_file. Got {type(f0_file)}")
        
        if hasattr(f0_file, "name") and not os.path.exists(f0_file.name):
            logger.warning("f0_file not found. Will use None instead.")
            f0_file = None
        
        if hasattr(index_file, "name"):
            index_file = index_file.name
        elif not isinstance(index_file, str) and index_file is not None:
            raise RuntimeError(f"pathlib.Path, str, or None expected for index_file. Got {type(index_file)}")
        
        if index_file is not None and not os.path.exists(index_file):
            logger.warning("index_file not found. Will use None instead.")
            index_file = None

        try:
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max
            times = {"npy": 0, "f0": 0, "infer": 0}

            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config, hubert_path)

            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                index_file,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            )

            tgt_sr = resample_sr if self.tgt_sr != resample_sr >= 16000 else self.tgt_sr

            return tgt_sr, audio_opt, times, None

        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return None, None, None, info

    def vc_multi(
        self,
        sid: int,
        paths: list[Path | str],
        opt_root: Path | str,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        f0_file: Path | str | None = None,
        index_file: Path | str | None = None,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        output_format: str = "wav",
        hubert_path: str | Path | None = None,
    ):
        if hasattr(opt_root, "name"):
            opt_root = opt_root.name

        try:
            os.makedirs(opt_root, exist_ok=True)
            paths = [path.name if hasattr(path, "name") else path for path in paths]
            infos = []
            for path in paths:
                tgt_sr, audio_opt, _, info = self.vc_inference(
                    sid,
                    path,
                    f0_up_key,
                    f0_method,
                    f0_file,
                    index_file,
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                    hubert_path,
                )
                if info:
                    try:
                        if output_format in ["wav", "flac"]:
                            sf.write(
                                f"{opt_root}/{os.path.basename(path)}.{output_format}",
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(
                                    f"{opt_root}/{os.path.basename(path)}.{output_format}",
                                    "wb",
                                ) as outf:
                                    wav2(wavf, outf, output_format)
                    except Exception:
                        info += traceback.format_exc()
                infos.append(f"{os.path.basename(path)}->{info}")
                yield "\n".join(infos)
            yield "\n".join(infos)
        except:
            yield traceback.format_exc()
