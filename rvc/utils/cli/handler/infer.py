import logging
from pathlib import Path

import click
from dotenv import load_dotenv
from scipy.io import wavfile



logging.getLogger("numba").setLevel(logging.WARNING)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="inference audio",
)
@click.option(
    "-m",
    "--modelPath",
    is_flag=False,
    type=str,
    help="Model path or filename (reads in the directory set in env)",
    required=True,
)
@click.option(
    "-i",
    "--inputPath",
    is_flag=False,
    type=Path,
    help="input audio path or folder",
    required=True,
)
@click.option(
    "-o",
    "--outputPath",
    is_flag=False,
    type=Path,
    help="output audio path or folder",
    required=True,
)
@click.option(
    "-s", "--sid", is_flag=False, type=int, help="Speaker/Singer id", default=0
)
@click.option("-fu", "--f0upkey", is_flag=False, type=int, help="Transpose", default=0)
@click.option(
    "-fm",
    "--f0method",
    is_flag=False,
    type=str,
    help="Pitch extraction algorith",
    default="rmvpe",
)
@click.option(
    "-ff", "--f0file", is_flag=False, type=Path, help="F0 curve file (optional)"
)
@click.option("-if", "--indexFile", is_flag=False, type=Path, help="Feature index file")
@click.option(
    "-ir",
    "--indexRate",
    is_flag=False,
    type=float,
    help="Search feature ratio",
    default=0.75,
)
@click.option(
    "-fr",
    "--filterRadius",
    is_flag=False,
    type=int,
    help="Apply median filtering",
    default=3,
)
@click.option(
    "-rsr",
    "--resamplesr",
    is_flag=False,
    type=int,
    help="Resample the output audio",
    default=0,
)
@click.option(
    "-rmr",
    "--rmsmixrate",
    is_flag=False,
    type=float,
    help="Adjust the volume envelope scaling",
    default=0.25,
)
@click.option(
    "-p",
    "--protect",
    is_flag=False,
    type=float,
    help="Protect voiceless consonants and breath sounds",
    default=0.33,
)
def infer(
    modelpath,
    inputpath,
    outputpath,
    sid,
    f0upkey,
    f0method,
    f0file,
    indexfile,
    indexrate,
    filterradius,
    resamplesr,
    rmsmixrate,
    protect,
):
    from rvc.modules.vc.modules import VC
    load_dotenv()
    vc = VC()
    vc.get_vc(modelpath)
    tgt_sr, audio_opt, times, _ = vc.vc_single(
        sid,
        inputpath,
        f0upkey,
        f0method,
        f0file,
        indexfile,
        indexrate,
        filterradius,
        resamplesr,
        rmsmixrate,
        protect,
    )
    if outputpath:
        wavfile.write(outputpath, tgt_sr, audio_opt)
        click.echo(times)
        click.echo(f"Finish inference. Check {outputpath}")
    return tgt_sr, audio_opt, times
