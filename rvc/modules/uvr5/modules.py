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
from rvc.modules.uvr5.vr import AudioPreprocess

logger: logging.Logger = logging.getLogger(__name__)


class UVR:
    def __init__(self):
        self.need_reformat: bool = True
        self.config: Config = Config()

    def uvr_wrapper(
        self,
        audio_path: Path,
        agg: int = 10,
        model_name: str | None = None,
        temp_dir: Path | None = None,
    ):
        infos = list()
        if model_name is None:
            model_name = os.path.basename(glob(f"{os.getenv('weight_uvr5_root')}/*")[0])

        pre_fun = AudioPreprocess(
            os.path.join(os.getenv("weight_uvr5_root"), model_name),  # + ".pth"
            int(agg),
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

        results = []

        for process_path in [process_paths]:
            print(f"path: {process_path}")
            info = sf.info(process_path)
            if not (info.channels == 2 and info.samplerate == "44100"):
                tmp_path = os.path.join(
                    temp_dir or os.environ.get("TEMP"), os.path.basename(process_path)
                )
                AudioSegment.from_file(process_path).export(
                    tmp_path,
                    format="wav",
                    codec="pcm_s16le",
                    bitrate="16k",
                    parameters=["-ar", "44100"],
                )

            results.append(
                pre_fun.process(
                    tmp_path or process_path,
                )
            )
            infos.append(f"{os.path.basename(process_path)}->Success")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
