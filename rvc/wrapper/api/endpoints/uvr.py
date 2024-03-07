from fastapi import APIRouter, Response, UploadFile, responses

from rvc.modules.uvr5.modules import UVR

router = APIRouter()


@router.post("/inference")
def uvr(inputpath, outputpath, modelname, format):
    uvr_module = UVR()
    uvr_module.uvr_wrapper(
        inputpath, outputpath, model_name=modelname, export_format=format
    )
    return responses.StreamingResponse(
        audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=inference.wav"},
    )
