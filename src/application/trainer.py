import torch
import torch.nn as nn

def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, output_path: str) -> None:
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )