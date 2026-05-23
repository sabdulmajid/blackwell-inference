#!/usr/bin/env python3
"""CPU-only static probe for vLLM MXFP4/NVFP4 backend selection.

This does not initialize CUDA or load a model. It imports the installed vLLM
Python selectors, replaces their platform/dependency probes with deterministic
fakes, and writes a JSON artifact that documents selection behavior.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
from typing import Any


class PatchStack:
    def __init__(self) -> None:
        self._items: list[tuple[Any, str, Any, bool]] = []

    def setattr(self, obj: Any, name: str, value: Any) -> None:
        existed = hasattr(obj, name)
        old = getattr(obj, name, None)
        self._items.append((obj, name, old, existed))
        setattr(obj, name, value)

    def restore(self) -> None:
        for obj, name, old, existed in reversed(self._items):
            if existed:
                setattr(obj, name, old)
            else:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
        self._items.clear()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def package_root(name: str) -> str | None:
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    return str(Path(dist.locate_file("")).resolve())


def enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def enum_value(value: Any) -> Any:
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, (str, int, float, bool, type(None))) else str(raw)


def make_fake_platform(device_capability_cls: Any, major: int, minor: int = 0) -> Any:
    capability = device_capability_cls(major, minor)

    class FakePlatform:
        @classmethod
        def is_cuda(cls) -> bool:
            return True

        @classmethod
        def is_xpu(cls) -> bool:
            return False

        @classmethod
        def is_rocm(cls) -> bool:
            return False

        @classmethod
        def is_cpu(cls) -> bool:
            return False

        @classmethod
        def get_device_capability(cls, device_id: int = 0) -> Any:
            return capability

        @classmethod
        def has_device_capability(
            cls, required: tuple[int, int] | int, device_id: int = 0
        ) -> bool:
            if isinstance(required, tuple):
                return capability >= device_capability_cls(*required)
            return capability.to_int() >= required

        @classmethod
        def is_device_capability(
            cls, required: tuple[int, int] | int, device_id: int = 0
        ) -> bool:
            if isinstance(required, tuple):
                return capability == device_capability_cls(*required)
            return capability.to_int() == required

        @classmethod
        def is_device_capability_family(cls, required: int, device_id: int = 0) -> bool:
            return (capability.to_int() // 10) == (required // 10)

    return FakePlatform


def evaluate_mxfp4_scenario(
    mxfp4: Any,
    fake_platform: Any,
    env_values: dict[str, bool],
    *,
    with_lora_support: bool,
    flashinfer_available: bool,
    triton_kernels_available: bool,
    torch_new_enough: bool,
) -> dict[str, Any]:
    patches = PatchStack()
    try:
        patches.setattr(mxfp4, "current_platform", fake_platform)
        patches.setattr(mxfp4, "has_flashinfer", lambda: flashinfer_available)
        patches.setattr(mxfp4, "has_triton_kernels", lambda: triton_kernels_available)
        patches.setattr(mxfp4, "is_torch_equal_or_newer", lambda _: torch_new_enough)
        for name, value in env_values.items():
            patches.setattr(mxfp4.envs, name, value)

        backend = mxfp4.get_mxfp4_backend(with_lora_support)
        capability = fake_platform.get_device_capability()
        return {
            "capability": [capability.major, capability.minor],
            "with_lora_support": with_lora_support,
            "flashinfer_available": flashinfer_available,
            "triton_kernels_available": triton_kernels_available,
            "torch_new_enough": torch_new_enough,
            "env": env_values,
            "backend_name": enum_name(backend),
            "backend_value": enum_value(backend),
            "is_device_capability_100": fake_platform.is_device_capability(100),
            "has_device_capability_100": fake_platform.has_device_capability(100),
            "is_device_capability_family_100": fake_platform.is_device_capability_family(100),
            "triton_range_sm90_to_before_sm110": (9, 0) <= capability < (11, 0),
        }
    finally:
        patches.restore()


def evaluate_nvfp4_helpers(
    flashinfer_fp4_moe: Any,
    fake_platform: Any,
    *,
    cutlass_available: bool,
    cutedsl_available: bool,
) -> dict[str, Any]:
    patches = PatchStack()
    try:
        patches.setattr(flashinfer_fp4_moe, "current_platform", fake_platform)
        patches.setattr(flashinfer_fp4_moe.envs, "VLLM_USE_FLASHINFER_MOE_FP4", True)
        patches.setattr(
            flashinfer_fp4_moe, "has_flashinfer_cutlass_fused_moe", lambda: cutlass_available
        )
        if hasattr(flashinfer_fp4_moe, "has_flashinfer_cutedsl_grouped_gemm_nt_masked"):
            patches.setattr(
                flashinfer_fp4_moe,
                "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
                lambda: cutedsl_available,
            )

        result = {
            "capability": list(fake_platform.get_device_capability()),
            "cutlass_available_probe": cutlass_available,
            "cutlass_helper_result": bool(
                flashinfer_fp4_moe.is_flashinfer_fp4_cutlass_moe_available()
            ),
        }
        if hasattr(flashinfer_fp4_moe, "is_flashinfer_fp4_cutedsl_moe_available"):
            result["cutedsl_available_probe"] = cutedsl_available
            result["cutedsl_helper_result"] = bool(
                flashinfer_fp4_moe.is_flashinfer_fp4_cutedsl_moe_available()
            )
        else:
            result["cutedsl_helper_result"] = "helper_absent_in_installed_vllm"
        return result
    finally:
        patches.restore()


def build_probe(upstream_head: str | None) -> tuple[int, dict[str, Any]]:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    data: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": timestamp,
        "run_status": "started",
        "probe_type": "static_no_cuda",
        "command": sys.argv,
        "upstream_head_observed": upstream_head,
        "packages": {
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "triton": package_version("triton"),
            "flashinfer_python": package_version("flashinfer-python"),
            "vllm_root": package_root("vllm"),
        },
    }

    if data["packages"]["vllm"] is None:
        data.update({"run_status": "skipped", "reason": "vllm is not installed"})
        return 0, data

    try:
        mxfp4 = importlib.import_module("vllm.model_executor.layers.quantization.mxfp4")
        interface = importlib.import_module("vllm.platforms.interface")
        flashinfer_fp4_moe = importlib.import_module(
            "vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe"
        )
    except Exception as exc:
        data.update({"run_status": "failed", "error": repr(exc)})
        return 1, data

    data["source_files"] = {
        "mxfp4": getattr(mxfp4, "__file__", None),
        "platform_interface": getattr(interface, "__file__", None),
        "flashinfer_fp4_moe": getattr(flashinfer_fp4_moe, "__file__", None),
    }

    device_capability_cls = interface.DeviceCapability
    platforms = {
        "sm90": make_fake_platform(device_capability_cls, 9, 0),
        "sm100": make_fake_platform(device_capability_cls, 10, 0),
        "sm120": make_fake_platform(device_capability_cls, 12, 0),
    }
    all_mxfp4_env = {
        "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": True,
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": True,
        "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS": True,
        "VLLM_MXFP4_USE_MARLIN": False,
    }
    base_env = {name: False for name in all_mxfp4_env}

    scenarios = {
        "sm90_flashinfer_bf16": evaluate_mxfp4_scenario(
            mxfp4,
            platforms["sm90"],
            {**base_env, "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16": True},
            with_lora_support=False,
            flashinfer_available=True,
            triton_kernels_available=True,
            torch_new_enough=True,
        ),
        "sm100_flashinfer_cutlass": evaluate_mxfp4_scenario(
            mxfp4,
            platforms["sm100"],
            {**base_env, "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS": True},
            with_lora_support=False,
            flashinfer_available=True,
            triton_kernels_available=True,
            torch_new_enough=True,
        ),
        "sm100_flashinfer_trtllm": evaluate_mxfp4_scenario(
            mxfp4,
            platforms["sm100"],
            {**base_env, "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": True},
            with_lora_support=False,
            flashinfer_available=True,
            triton_kernels_available=True,
            torch_new_enough=True,
        ),
        "sm120_flashinfer_all_flags": evaluate_mxfp4_scenario(
            mxfp4,
            platforms["sm120"],
            all_mxfp4_env,
            with_lora_support=False,
            flashinfer_available=True,
            triton_kernels_available=True,
            torch_new_enough=True,
        ),
        "sm120_lora_flashinfer_all_flags": evaluate_mxfp4_scenario(
            mxfp4,
            platforms["sm120"],
            all_mxfp4_env,
            with_lora_support=True,
            flashinfer_available=True,
            triton_kernels_available=True,
            torch_new_enough=True,
        ),
        "sm120_no_flashinfer_triton_available": evaluate_mxfp4_scenario(
            mxfp4,
            platforms["sm120"],
            base_env,
            with_lora_support=False,
            flashinfer_available=False,
            triton_kernels_available=True,
            torch_new_enough=True,
        ),
    }

    data["mxfp4_backend_scenarios"] = scenarios
    data["nvfp4_flashinfer_helper_scenarios"] = {
        "sm120_cutlass_and_cutedsl_available": evaluate_nvfp4_helpers(
            flashinfer_fp4_moe,
            platforms["sm120"],
            cutlass_available=True,
            cutedsl_available=True,
        ),
        "sm100_cutlass_and_cutedsl_available": evaluate_nvfp4_helpers(
            flashinfer_fp4_moe,
            platforms["sm100"],
            cutlass_available=True,
            cutedsl_available=True,
        ),
    }
    data["run_status"] = "completed"
    return 0, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="JSON artifact path.")
    parser.add_argument("--upstream-head", default=None, help="Optional upstream vLLM HEAD SHA.")
    args = parser.parse_args()

    rc, data = build_probe(args.upstream_head)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "run_status": data["run_status"]}, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
