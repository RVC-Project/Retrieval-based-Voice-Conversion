# -*- coding: utf-8 -*-
import numpy as np, librosa as rosa, ffmpeg
##
def wavread(
    filepath:str, fs:int=0, ch:int=0,
    dtype=None, read_async=False, res_type:str="soxr_hq",
  **kwargs) -> tuple[np.ndarray, int]:
    """
    Reads an audio file using `ffmpeg` &
        returns the audio data as an `np.ndarray` &
        the sample rate (`fs`).
    Args:
        filepath (str): Path to the audio file.
        fs (int, optional): Desired sample rate. If 0,
            the original sample rate is used. Defaults to 0.
        ch (int, optional): Desired number of channels. If 0,
            the original number of channels is used. Defaults to 0.
        dtype (data-type, optional): Desired data type
            for the output array. If None, the data is returned
            as 32-bit float. Defaults to None.
        read_async (bool, optional): If True, reads the audio data
            asynchronously. Defaults to False.
        res_type (str, optional): Resampling type
            defined in `librosa`, defaulting to "soxr_hq".
        **kwargs: Additional arguments.
    Returns:
        tuple[np.ndarray, int]: A tuple containing the audio data
            as a NumPy array and the sample rate.
    """
    ## Performing FFProbe the Audio File to get Stream Information
    d_probe = ffmpeg.probe(filepath)
    st_audio = next(
        s for s in d_probe["streams"] \
        if s["codec_type"] == "audio")
    ch_origin = int(st_audio["channels"])
    fs_origin = int(st_audio["sample_rate"])
    ch = ch or ch_origin
    fs = fs or fs_origin
    ## Determining the float32 format based on system endianness
    fp32, ffmpeg_format = ("<f4", "f32le") if np.little_endian else \
                          (">f4", "f32be")
    ffmpeg_acodec = f"pcm_{ffmpeg_format}"
    ## Setting Keyword-based (non-positional) Args of `ffmpeg.output`
    kwgs_output = {
        "format": ffmpeg_format,
        "acodec": ffmpeg_acodec,
        ## Using original sample rate and channels if not specified
        "ac": ch,
        #"ar": fs, # Resampling later using `librosa`
        "loglevel": "error"}
    ## Reading the Audio Asynchronously
    if read_async:
        async_pipe = (
            ffmpeg
            .input(filepath)
            .output("pipe:", **kwgs_output)
            .run_async(pipe_stdout=True))
        pcm_raw = async_pipe.stdout.read()
        async_pipe.wait()
    ## Reading the Audio Synchronously
    else:
        pcm_raw, _ = (
            ffmpeg
            .input(filepath)
            .output("pipe:", **kwgs_output)
            .run(
                capture_stdout = True,
                capture_stderr = True,
                ))
    ## Converting the Raw PCM Data to Float32 NumPy Array
    x_raw = np.frombuffer(pcm_raw, dtype=fp32).reshape(-1, ch).T
    ## Resampling using `librosa`, if necessary
    x_res = rosa.resample(x_raw, orig_sr=fs_origin, target_sr=fs,
        res_type=res_type, axis=-1) if fs != fs_origin else x_raw
    ## Converting to the target data type if specified
    if dtype:
        dt = np.dtype(dtype)
        if dt.kind == "i":
            x_res = np.clip(x_res, -1., +1.)
            x_res = (x_res * (np.iinfo(dt).max - 1)).astype(dt)
    ## Returning with shapped as (channels, samples)
    return x_res, fs
##
def wavread_rosa(filepath:str,
    fs:int=22050, mono:bool=True,
    dtype=np.float32, res_type:str="soxr_hq",
  **kwargs) -> tuple[np.ndarray, int]:
    """
    Reads an audio file using `ffmpeg` &
        returns the audio data as an `np.ndarray` &
        the sample rate (`fs`).
    Args:
        filepath (str): Path to the audio file.
        fs (int, optional): Desired sample rate. Defaults to 22050.
        mono (bool, optional): If True, converts the audio to mono.
            Defaults to True.
        dtype (data-type, optional): Desired data type
            for the output array. If None, the data is returned
            as 32-bit float. Defaults to `np.float32`.
        res_type (str, optional): Resampling type
            defined in `librosa`, defaulting to "soxr_hq".
        **kwargs: Additional arguments.
    Returns:
        tuple[np.ndarray, int]: A tuple containing the audio data
            as a NumPy array and the sample rate.
    """
    x, fs = wavread(filepath,
        fs = fs,
        ch = 1 if mono else 2,
        dtype = dtype,
        res_type = res_type,
      **kwargs)
    if x.shape[0] < 2:
        x = x.squeeze(0)
    return x, fs
##
