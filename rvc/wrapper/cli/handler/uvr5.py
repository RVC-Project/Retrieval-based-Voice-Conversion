from pathlib import Path

import click

from rvc.modules.uvr5.modules import UVR


@click.command()
@click.option(
    "-m",
    "--modelName",
    is_flag=False,
    type=str,
    help="Model path or filename (reads in the directory set in env)",
    # required=True,
)
@click.option(
    "-i",
    "--inputPath",
    is_flag=False,
    type=Path,
    help="input audio path or folder",
    # required=True,
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
    "-f",
    "--format",
    is_flag=False,
    type=str,
    help="output Format",
)
def uvr(modelname, inputpath, outputpath, format):
    uvr_module = UVR()
    uvr_module.uvr_wrapper(
        inputpath, outputpath, model_name=modelname, export_format=format
    )
    click.echo(f"Finish uvr5. Check {outputpath}")
