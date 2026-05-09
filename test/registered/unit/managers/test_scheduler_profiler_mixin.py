import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_scheduler_profiler_mixin():
    class FakeProfileReqOutput:
        def __init__(self, success, message):
            self.success = success
            self.message = message

    fake_torch = types.ModuleType("torch")
    fake_torch.profiler = types.SimpleNamespace(
        ProfilerActivity=types.SimpleNamespace(CPU="cpu", CUDA="cuda"),
        profile=lambda **kwargs: None,
    )
    fake_torch.cuda = types.SimpleNamespace(
        memory=types.SimpleNamespace(
            _record_memory_history=lambda **kwargs: None,
            _dump_snapshot=lambda path: None,
        ),
        cudart=lambda: types.SimpleNamespace(
            cudaProfilerStart=lambda: None,
            cudaProfilerStop=lambda: None,
        ),
    )
    fake_torch.distributed = types.SimpleNamespace(barrier=lambda group: None)

    stub_modules = {
        "torch": fake_torch,
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.environ": types.SimpleNamespace(
            envs=types.SimpleNamespace(
                SGLANG_PROFILE_V2=types.SimpleNamespace(get=lambda: False)
            )
        ),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.io_struct": types.SimpleNamespace(
            ProfileReq=object,
            ProfileReqOutput=FakeProfileReqOutput,
            ProfileReqType=object,
        ),
        "sglang.srt.model_executor": types.ModuleType("sglang.srt.model_executor"),
        "sglang.srt.model_executor.forward_batch_info": types.SimpleNamespace(
            ForwardMode=object
        ),
        "sglang.srt.server_args": types.SimpleNamespace(
            get_global_server_args=lambda: types.SimpleNamespace(base_gpu_id=0)
        ),
        "sglang.srt.utils": types.SimpleNamespace(is_npu=lambda: False),
        "sglang.srt.utils.profile_merger": types.SimpleNamespace(ProfileMerger=object),
        "sglang.srt.utils.profile_utils": types.SimpleNamespace(ProfileManager=object),
    }

    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/managers/scheduler_profiler_mixin.py"
    )
    spec = importlib.util.spec_from_file_location(
        "scheduler_profiler_mixin_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stub_modules):
        spec.loader.exec_module(module)
    return module


class FakeTorchProfiler:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True


class TestSchedulerProfilerMixin(unittest.TestCase):
    def test_npu_start_profile_passes_experimental_config(self):
        scheduler_profiler_mixin = load_scheduler_profiler_mixin()
        created_configs = []

        class FakeExperimentalConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created_configs.append(self)

        fake_torch_npu = types.SimpleNamespace(
            profiler=types.SimpleNamespace(
                _ExperimentalConfig=FakeExperimentalConfig,
                ExportType=types.SimpleNamespace(Text="text"),
                ProfilerLevel=types.SimpleNamespace(Level1="level1"),
                tensorboard_trace_handler=lambda output_dir: ("handler", output_dir),
            )
        )
        fake_profile = FakeTorchProfiler()
        profile_kwargs = {}

        def profile(**kwargs):
            profile_kwargs.update(kwargs)
            return fake_profile

        scheduler = types.SimpleNamespace(
            profiler_activities=["CPU", "GPU"],
            torch_profiler_with_stack=None,
            torch_profiler_record_shapes=None,
            torch_profiler_output_dir=Path("/tmp/profile"),
            profile_id="test-profile",
            tp_rank=0,
            profile_in_progress=False,
            torch_profiler=None,
        )

        with (
            patch.object(scheduler_profiler_mixin, "_is_npu", True),
            patch.object(
                scheduler_profiler_mixin, "torch_npu", fake_torch_npu, create=True
            ),
            patch.object(scheduler_profiler_mixin.torch.profiler, "profile", profile),
        ):
            scheduler_profiler_mixin.SchedulerProfilerMixin.start_profile(scheduler)

        self.assertTrue(fake_profile.started)
        self.assertIs(profile_kwargs["experimental_config"], created_configs[0])
        self.assertEqual(
            created_configs[0].kwargs,
            {
                "export_type": ["text"],
                "profiler_level": "level1",
            },
        )


if __name__ == "__main__":
    unittest.main()
