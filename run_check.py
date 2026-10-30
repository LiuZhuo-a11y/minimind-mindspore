import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np


def main():
    """Minimal GPU sanity check.

    Runs a single 2x2 matmul to confirm MindSpore can actually reach the
    device. If this fails, the problem is the driver / CUDA / MindSpore build,
    not this repository.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    x = Tensor(np.ones((2, 2), dtype=np.float32))
    y = Tensor(np.ones((2, 2), dtype=np.float32))
    z = ops.matmul(x, y)
    print("MindSpore version:", ms.__version__)
    print("Result:", z.asnumpy())
    print("MindSpore GPU check OK")


if __name__ == "__main__":
    main()
