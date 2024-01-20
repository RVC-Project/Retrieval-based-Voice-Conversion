import torch

from rvc.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


def export_onnx(ModelPath, ExportedPath):
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768

    test_phone = torch.rand(1, 200, vec_channels)  # hidden unit
    test_phone_lengths = torch.tensor(
        [200]
    ).long()  # hidden unit length (doesn't seem to help)）
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # Base frequency (in Hz)
    test_pitchf = torch.rand(1, 200)  # nsf base frequency
    test_ds = torch.LongTensor([0])  # Speaker ID
    test_rnd = torch.rand(1, 192, 200)  # Noise (add random factor)

    device = "cpu"  # Device on export (does not affect use of model)

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False, version=cpt.get("version", "v1")
    )  # fp32 export (C++ has to manually rearrange memory to support fp16 so no fp16 for now)
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker) Multi-Role Mixed Track Export
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    return "Finished"
