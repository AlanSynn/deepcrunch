import torch


# Loading the PyTorch model and running the dynamic shape inference
def load_model(model_path: str, device: str = "cpu"):
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model


def convert_torch_to_onnx(
    model_path, input_shape, device="cpu", output_path="test.onnx", opset_version=11
):
    model = load_model(model_path, device)

    with torch.no_grad():
        input_names = ["input_0"]
        output_names = ["output_0"]
        inputs = torch.randn(input_shape).to(device)

        dynamic_axes = {"input_0": {0: "batch_size"}, "output_0": {0: "batch_size"}}

        torch_out = torch.onnx._export(
            model,
            inputs,
            output_path,
            export_params=True,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
        )
        return torch_out
