import time

import numpy as np
import onnxruntime as ort
from tqdm import tqdm, trange


def do_inference(
    model_path,
    output_string,
    input_dim=(1, 3, 224, 224),
    providers=["CPUExecutionProvider"],
):
    origin_ort_session = ort.InferenceSession(model_path, providers=providers)

    warm_up_count = 5
    for i in range(warm_up_count):
        x = np.random.rand(*input_dim).astype("float32")
        origin_ort_session.run(None, {origin_ort_session._inputs_meta[0].name: x})

    count = 100
    start_total = time.time()

    with trange(count) as t:
        for i in t:
            x = np.random.rand(*input_dim).astype("float32")
            start = time.time()
            origin_ort_session.run(None, {origin_ort_session._inputs_meta[0].name: x})
            t.set_postfix(time=f"{(time.time() - start)*1000:0.2f}" + "ms")

    end_total = time.time()
    print(f"{output_string} Average Time: {(end_total - start_total)*1000/count:.3f}")
