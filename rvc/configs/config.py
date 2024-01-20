import argparse
import json
import logging
import os
import sys
from multiprocessing import cpu_count

import torch

try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        from rvc.lib.ipex import ipex_init

        ipex_init()
except (ImportError, Exception):
    pass

logger: logging.Logger = logging.getLogger(__name__)


version_config_list: list = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__)))
    for file in files
    if file.endswith(".json")
]


class Config:
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.device: str = "cuda:0"
        self.is_half: bool = True
        self.use_jit: bool = False
        self.n_cpu: int = cpu_count()
        self.gpu_name: str | None = None
        self.json_config = self.load_config_json()
        self.gpu_mem: int | None = None
        self.instead: str | None = None
        (
            self.python_cmd,
            self.listen_port,
            self.noparallel,
            self.noautoopen,
            self.dml,
        ) = self.arg_parse()
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        return {
            config_file: json.load(open(config_file, "r"))
            for config_file in version_config_list
        }

    @staticmethod
    def arg_parse() -> tuple:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument(
            "--pycmd",
            type=str,
            default=sys.executable or "python",
            help="Python command",
        )
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )
        cmd_opts: argparse.Namespace
        cmd_opts, _ = parser.parse_known_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
        )

    @staticmethod
    def has_mps() -> bool:
        return torch.backends.mps.is_available() and not torch.zeros(1).to(
            torch.device("mps")
        )

    @staticmethod
    def has_xpu() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def use_fp32_config(self) -> None:
        for config_file, data in self.json_config.items():
            try:
                data["train"]["fp16_run"] = False
                with open(config_file, "w") as json_file:
                    json.dump(data, json_file, indent=4)
            except Exception as e:
                logger.info(f"Error updating {config_file}: {str(e)}")
        logger.info("overwrite configs.json")

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info(f"Found GPU {self.gpu_name}, force to fp32")
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info(f"Found GPU {self.gpu_name}")
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        elif self.dml:
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        elif self.is_half:
            # 6G PU_RAM conf
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G GPU_RAM conf
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        logger.info(f"Use {self.dml or self.instead} instead")
        logger.info(f"is_half:{self.is_half}, device:{self.device}")
        return x_pad, x_query, x_center, x_max
