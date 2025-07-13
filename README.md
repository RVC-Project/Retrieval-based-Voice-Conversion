<div align="center">

<h1>Retrieval-based-Voice-Conversion</h1>
An easy-to-use Voice Conversion framework based on VITS.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>

[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion/blob/develop/LICENSE)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------


> [!NOTE]
> Currently under development... Provided as a library and API in rvc

## Installation and usage

### Install rvc using pip
```sh
pip install git+https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
```

### Standard Setup

First, create a directory in your project. The `assets` folder will contain the models needed for inference and training, and the `result` folder will contain the results of the training.

```sh
rvc init
```
This will create an `assets` folder and `.env` in your working directory.

> [!WARNING]
> The directory should be empty or without an assets folder.

### Custom Setup

If you have already downloaded models or want to change these configurations, edit the `.env` file.
If you do not already have a `.env` file,

```sh
rvc env create
```
can create one.

Also, when downloading a model, you can use the

```sh
rvc dlmodel
```
or
```
rvc dlmodel {download_dir}
```

Finally, specify the location of the model in the env file, and you are done!



### Library Usage

#### Inference Audio
```python
from pathlib import Path

from dotenv import load_dotenv
from scipy.io import wavfile

from rvc.modules.vc.modules import VC


def main():
      vc = VC()
      vc.get_vc("{model.pth}")
      tgt_sr, audio_opt, times, _ = vc.vc_inference(
            1, Path("{InputAudio}")
      )
      wavfile.write("{OutputAudio}", tgt_sr, audio_opt)


if __name__ == "__main__":
      load_dotenv("{envPath}")
      main()

```

### CLI Usage

#### Inference Audio

```sh
rvc infer -m {model.pth} -i {input.wav} -o {output.wav}
```

| option        | flag&nbsp; | type         | default value | description                                                                                                                                                                                                                                    |
|---------------|------------|--------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| modelPath     | -m         | Path         | *required     | Model path or filename (reads in the directory set in env)                                                                                                                                                                                     |
| inputPath     | -i         | Path         | *required     | Input audio path or folder                                                                                                                                                                                                                     |
| outputPath    | -o         | Path         | *required     | Output audio path or folder                                                                                                                                                                                                                    |
| sid           | -s         | int          | 0             | Speaker/Singer ID                                                                                                                                                                                                                              |
| f0_up_key     | -fu        | int          | 0             | Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)                                                                                                                                                      |
| f0_method     | -fm        | str          | rmvpe         | pitch extraction algorithm (pm, harvest, crepe, rmvpe                                                                                                                                                                                          |
| f0_file       | -ff        | Path \| None | None          | F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation                                                                                                                                                     |
| index_file    | -if        | Path \| None | None          | Path to the feature index file                                                                                                                                                                                                                 |
| index_rate    | -if        | float        | 0.75          | Search feature ratio (controls accent strength, too high has artifacting)                                                                                                                                                                      |
| filter_radius | -fr        | int          | 3             | If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness                                                                                                               |
| resample_sr   | -rsr       | int          | 0             | Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling                                                                                                                                              |
| rms_mix_rate  | -rmr       | float        | 0.25          | Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume |
| protect       | -p         | float        | 0.33          | Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy                                 |

### API Usage
First, start up the server.
```sh
rvc-api
```
or
```sh
poetry run poe rvc-api
```

#### Inference Audio

##### Get as blob
```sh
curl -X 'POST' \
      'http://127.0.0.1:8000/inference?res_type=blob' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'modelpath={model.pth}' \
      -F 'input={input audio path}'
```

##### Get as json(include time)
```sh
curl -X 'POST' \
      'http://127.0.0.1:8000/inference?res_type=json' \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'modelpath={model.pth}' \
      -F 'input={input audio path}'
```

### Docker Usage

Build and run via script:

```bash
./docker-run.sh
```

**Or** use manually:

1. Build:

   ```bash
   docker build -t "rvc" .
   ```

2. Run:

   ```bash
   docker run -it \
     -p 8000:8000 \
     -v "${PWD}/assets/weights:/weights:ro" \
     -v "${PWD}/assets/indices:/indices:ro" \
     -v "${PWD}/assets/audios:/audios:ro" \
     "rvc"
   ```

Notice assumption that weights, indices and input audios are stored in `current-directory/assets`
