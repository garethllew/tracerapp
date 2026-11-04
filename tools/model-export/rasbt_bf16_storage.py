"""Store the rasbt ModernBERT weights as bfloat16 in the ONNX file (half the
size), cast to float32 at load. Compute stays FP32, so there's no
quantisation: the model was trained in bf16, so its weights ARE bf16 values
and truncating their FP32 copies back to bf16 is lossless.

Every float initializer with >= 1024 elements becomes a BFLOAT16 initializer
followed by Cast(to=FLOAT) producing the original name. ONNX Runtime folds
these casts once at session load, so inference speed is unchanged.

Usage:  python -I rasbt_bf16_storage.py   (writes work/onnx/rasbt-bf16w.onnx)
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_rasbt import FP32, OUT  # noqa: E402

DST = OUT / "rasbt-bf16w.onnx"


def main():
    m = onnx.load(str(FP32))
    g = m.graph
    new_inits, casts, lossy, kept = [], [], 0, 0
    for init in list(g.initializer):
        if init.data_type != TensorProto.FLOAT or np.prod(init.dims) < 1024:
            new_inits.append(init)
            kept += 1
            continue
        a = numpy_helper.to_array(init).astype(np.float32)
        bits = a.view(np.uint32)
        if np.any(bits & 0xFFFF):
            lossy += 1  # not exactly bf16: round-to-nearest-even instead of truncating
            bits = bits + 0x7FFF + ((bits >> 16) & 1)
        bf = (bits >> 16).astype(np.uint16)
        t = TensorProto()
        t.name = init.name + "__bf16"
        t.data_type = TensorProto.BFLOAT16
        t.dims.extend(init.dims)
        t.raw_data = bf.tobytes()
        new_inits.append(t)
        casts.append(helper.make_node("Cast", [t.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
    del g.initializer[:]
    g.initializer.extend(new_inits)
    for n in reversed(casts):
        g.node.insert(0, n)
    onnx.save_model(m, str(DST), save_as_external_data=False)
    onnx.checker.check_model(str(DST))
    print(f"{len(casts)} weight tensors stored as bf16 ({lossy} were not exactly bf16, rounded), {kept} small kept FP32; "
          f"{DST.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
