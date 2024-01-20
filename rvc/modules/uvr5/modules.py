import logging
import os
import traceback
from glob import glob
from pathlib import Path

import soundfile as sf
import torch
from pydub import AudioSegment

from rvc.configs.config import Config
from rvc.modules.uvr5.mdxnet import MDXNetDereverb
from rvc.modules.uvr5.vr import AudioPre, AudioPreDeEcho

logger: logging.Logger = logging.getLogger(__name__)


class UVR:
    def __init__(self):
        self.need_reformat: bool = True
        self.config: Config = Config()

    def uvr_wrapper(
        self,
        audio_path: Path,
        save_vocal_path: Path | None = None,
        save_ins_path: Path | None = None,
        agg: int = 10,
        export_format: str = "flac",
        model_name: str | None = None,
        temp_path: Path | None = None,
    ):
        infos = []
        save_vocal_path = (
            os.getenv("save_uvr_path") if not save_vocal_path else save_vocal_path
        )
        save_ins_path = (
            os.getenv("save_uvr_path") if not save_ins_path else save_ins_path
        )

        if model_name is None:
            model_name = os.path.basename(glob(f"{os.getenv('weight_uvr5_root')}/*")[0])
        is_hp3 = "HP3" in model_name

        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, self.config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name  # + ".pth"
                ),
                device=self.config.device,
                is_half=self.config.is_half,
            )

        process_paths = (
            [
                _
                for _ in glob(f"{audio_path}/*")
                if os.path.splitext(_)[-1][1:].upper() in sf.available_formats()
            ]
            if os.path.isdir(audio_path)
            else audio_path
        )

        for process_path in [process_paths]:
            print(f"path: {process_path}")
            info = sf.info(process_path)
            if not (info.channels == 2 and info.samplerate == "44100"):
                tmp_path = os.path.join(
                    temp_path or os.environ.get("TEMP"), os.path.basename(process_path)
                )
                AudioSegment.from_file(process_path).export(
                    tmp_path,
                    format="wav",
                    codec="pcm_s16le",
                    bitrate="16k",
                    parameters=["-ar", "44100"],
                )

            pre_fun._path_audio_(
                process_path,
                save_vocal_path,
                save_ins_path,
                export_format,
                is_hp3=is_hp3,
            )
            infos.append(f"{os.path.basename(process_path)}->Success")
            yield "\n".join(infos)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
