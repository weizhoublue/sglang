import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MHC_HEAD_PATH = _REPO_ROOT / "python/sglang/srt/layers/mhc_head.py"


class _FakeDType:
    def __init__(self, name):
        self.name = name
        self.element_ty = self


class _FakeTensor:
    def __init__(self, shape, *, dtype, device="cuda"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def is_contiguous(self):
        return True

    def dim(self):
        return len(self.shape)

    def numel(self):
        n = 1
        for dim in self.shape:
            n *= dim
        return n


class _FakeKernel:
    def __init__(self, fn):
        self.fn = fn
        self.launches = []

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return _launch


def _load_mhc_head_with_fake_deps():
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = _FakeDType("float32")
    fake_torch.empty = lambda shape, *, dtype, device: _FakeTensor(
        shape, dtype=dtype, device=device
    )
    fake_torch.Tensor = _FakeTensor

    fake_triton = types.ModuleType("triton")
    fake_triton.jit = lambda fn: _FakeKernel(fn)
    fake_triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()

    fake_tl = types.ModuleType("triton.language")
    fake_tl.constexpr = object()

    with mock.patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "triton": fake_triton,
            "triton.language": fake_tl,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "_mhc_head_under_test", _MHC_HEAD_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestMhcHead(unittest.TestCase):
    def test_non_power_of_two_hc_mult_launches_with_real_hc_mult(self):
        module = _load_mhc_head_with_fake_deps()

        x_dtype = _FakeDType("float16")
        x = _FakeTensor((2, 3, 5), dtype=x_dtype)
        hc_fn = _FakeTensor((3, 15), dtype=module.torch.float32)
        hc_scale = _FakeTensor((1,), dtype=module.torch.float32)
        hc_base = _FakeTensor((3,), dtype=module.torch.float32)

        module.fused_hc_head(x, hc_fn, hc_scale, hc_base, 1e-6, 1e-3)

        _grid, _args, kwargs = module._hc_head_kernel.launches[-1]
        self.assertEqual(kwargs["HC_MULT"], 4)
        self.assertEqual(kwargs["HC_MULT_REAL"], 3)


if __name__ == "__main__":
    unittest.main()
