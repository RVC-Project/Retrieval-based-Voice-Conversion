# -*- coding: utf-8 -*-
import numpy as np, ffmpeg
##
def wavread(
    filepath:str, fs:int=0, ch:int=0,
    dtype=None, read_async=False,
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
    ## Using original sample rate and channels if not specified
    fs = fs or int(st_audio["sample_rate"])
    ch = ch or int(st_audio["channels"])
    ## Determining the float32 format based on system endianness
    fp32 = "<f4" if np.little_endian else ">f4"
    ffmpeg_format = "f32le" if np.little_endian else "f32be"
    ffmpeg_acodec = f"pcm_{ffmpeg_format}"
    ## Reading the Audio Asynchronously
    if read_async:
        async_pipe = (
            ffmpeg
            .input(filepath)
            .output("pipe:",
                format = ffmpeg_format,
                acodec = ffmpeg_acodec,
                ac = ch,
                ar = fs,
                loglevel = "error")
            .run_async(pipe_stdout=True))
        pcm_raw = async_pipe.stdout.read()
        async_pipe.wait()
        x_raw = np.frombuffer(pcm_raw, dtype=fp32)
    ## Reading the Audio Synchronously
    else:
        pcm_out, _ = (
            ffmpeg
            .input(filepath)
            .output("pipe:",
                format = ffmpeg_format,
                acodec = ffmpeg_acodec,
                ac = ch,
                ar = fs,
                loglevel = "error")
            .run(
                capture_stdout = True,
                capture_stderr = True,
                ))
        x_raw = np.frombuffer(pcm_out, dtype=fp32)
    ## Converting to the target data type if specified
    if dtype:
        dt = np.dtype(dtype)
        if dt.kind == "i":
            x_raw = np.clip(x_raw, -1., +1.)
            x_raw = (x_raw * (np.iinfo(dt).max - 1)).astype(dt)
    ## Returning with shapped as (channels, samples)
    return x_raw.reshape(-1, ch).T, fs
##
