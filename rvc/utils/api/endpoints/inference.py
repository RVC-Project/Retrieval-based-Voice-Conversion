from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, UploadFile, Response, responses
from pydantic import BaseModel
from scipy.io import wavfile

from rvc.modules.vc.modules import VC

router = APIRouter()


@router.post("/inference")
def inference(
    modelpath: str | UploadFile,
    input: Path | UploadFile,
    sid: int = 0,
    f0_up_key: int = 0,
    f0_method: str = "rmvpe",
    f0_file: Path | None = None,
    index_file: Path | None = None,
    index_rate: float = 0.75,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
):
    vc = VC()
    vc.get_vc(modelpath)
    tgt_sr, audio_opt, times, _ = vc.vc_single(
        sid,
        input,
        f0_up_key,
        f0_method,
        f0_file,
        index_file,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    )
    wavfile.write(wv := BytesIO(), tgt_sr, audio_opt)
    print(times)
    return responses.StreamingResponse(
        wv,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=inference.wav"},
    )
