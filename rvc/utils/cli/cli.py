import re
from typing import Optional, Pattern

import click

from rvc.utils.cli.handler.infer import infer
from rvc.utils.cli.handler.train import train
from rvc.utils.cli.handler.uvr5 import uvr
from rvc.utils.cli.utils.dlmodel import dlmodel
from rvc.utils.cli.utils.env import env
from rvc.utils.cli.utils.initialize import init


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="rvc cli feature list",
)
def cli():
    pass


def main():
    cli.add_command(infer)
    cli.add_command(train)
    cli.add_command(uvr)
    cli.add_command(dlmodel)
    cli.add_command(env)
    cli.add_command(init)
    cli()


if __name__ == "__main__":
    main()
