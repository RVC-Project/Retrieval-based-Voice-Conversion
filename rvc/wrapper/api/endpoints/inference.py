import json
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Response, UploadFile, Body, responses, Form, Query
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from scipy.io import wavfile
from base64 import b64encode
from rvc.modules.vc.modules import VC
import glob
import os

router = APIRouter()
from dotenv import load_dotenv

load_dotenv()


@router.post("/inference")
def inference(
    input_audio: Path | UploadFile,
    modelpath: Path
    | UploadFile = Body(
        ...,
        enum=[
            os.path.basename(file)
            for file in glob.glob(f"{os.getenv('weight_root')}/*")
        ],
    ),
    res_type: str = Query("blob", enum=["blob", "json"]),
    sid: int = 0,
    f0_up_key: int = 0,
    f0_method: str = Query(
        "rmvpe", enum=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"]
    ),
    f0_file: Path | None = None,
    index_file: Path | None = None,
    index_rate: float = 0.75,
    filter_radius: int = 3,
    resample_sr: int = 0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
):
    print(res_type)
    vc = VC()
    vc.get_vc(modelpath)
    tgt_sr, audio_opt, times, _ = vc.vc_inference(
        sid,
        input_audio,
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
    if res_type == "blob":
        return responses.StreamingResponse(
            wv,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=inference.wav"},
        )
    else:
        return JSONResponse(
            {
                "time": json.loads(json.dumps(times)),
                "audio": b64encode(wv.read()).decode("utf-8"),
            }
        )
