"""ExecuTorch export + on-device runner helpers (XNNPACK CPU and QNN HTP on QCS8275 / V75)."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ET_ROOT = Path("/home/arduino/QualcommDeveloper/executorch")
XNN_RUNNER = ET_ROOT / "build-xnnpack" / "executor_runner"
QNN_RUNNER = ET_ROOT / "examples" / "qualcomm" / "executor_runner" / "qnn_executor_runner"
QNN_BACKEND_LIB_DIR = ET_ROOT / "build-x86" / "backends" / "qualcomm"  # aarch64 despite the dir name
SOC = "QCS8275"

logging.getLogger().setLevel(logging.WARNING)


def export_xnnpack(model: torch.nn.Module, example: tuple[torch.Tensor, ...], out: Path) -> Path:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    ep = torch.export.export(model.eval(), example, strict=False)
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    out.write_bytes(prog.buffer)
    return out


def export_qnn(model: torch.nn.Module, example: tuple[torch.Tensor, ...], calibration: Sequence[torch.Tensor],
               out: Path, quant: str = "16a8w") -> Path:
    from executorch.backends.qualcomm.export_utils import make_quantizer
    from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from torchao.quantization.pt2e import MinMaxObserver
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    # Calibration runs at batch 1 and some heads are scalars (hand flag, handedness). The default
    # moving-average observer collapses their min/max toward the running mean, clipping them to ~0.5
    # after the sigmoid, so track the true extremes instead.
    quantizer = make_quantizer(quant_dtype=getattr(QuantDtype, f"use_{quant}"), soc_model=SOC,
                               act_observer=MinMaxObserver)
    captured = torch.export.export(model.eval(), example, strict=False).module()
    annotated = prepare_pt2e(captured, quantizer)
    with torch.no_grad():
        for x in calibration:
            annotated(x.unsqueeze(0) if x.dim() == 3 else x)
    quantized = convert_pt2e(annotated)

    specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, SOC),
        backend_options=generate_htp_compiler_spec(use_fp16=False),
    )
    edge = to_edge_transform_and_lower_to_qnn(quantized, example, specs)
    prog = edge.to_executorch(config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=True, alloc_graph_output=True),
        segment_alignment=get_qnn_context_binary_alignment(),
    ))
    out.write_bytes(prog.buffer)
    return out


def delegation_summary(pte: Path) -> dict:
    """Count delegated vs CPU-fallback ops by inspecting the serialized program."""
    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(pte.read_bytes()).program
    plan = program.execution_plan[0]
    ops = [plan.operators[c.op_index].name for c in (ch.instr_args for ch in plan.chains[0].instructions)
           if hasattr(c, "op_index")]
    delegates = [d.id for d in plan.delegates]
    return {"delegates": delegates, "cpu_ops": sorted(set(ops)), "num_cpu_op_calls": len(ops)}


def _write_inputs(tmp: Path, inputs: Sequence[np.ndarray]) -> list[Path]:
    paths = []
    for i, x in enumerate(inputs):
        p = tmp / f"input_{i}.raw"
        np.ascontiguousarray(x, dtype=np.float32).tofile(p)
        paths.append(p)
    return paths


def run_xnnpack(pte: Path, inputs: Sequence[np.ndarray], iterations: int = 20,
                cpu_threads: int = 4) -> tuple[list[np.ndarray], float]:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = _write_inputs(tmp, inputs)
        cmd = [str(XNN_RUNNER), f"--model_path={pte}", f"--inputs={','.join(map(str, paths))}",
               f"--output_file={tmp / 'out'}", "--print_output=none", f"--num_executions={iterations}",
               f"--cpu_threads={cpu_threads}"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log = res.stdout + res.stderr
        m = re.search(r"executed successfully (\d+) time\(s\) in ([\d.]+) ms", log)
        ms = float(m.group(2)) / int(m.group(1)) if m else float("nan")
        outs = [np.fromfile(p, dtype=np.float32) for p in sorted(tmp.glob("out-*.bin"), key=lambda p: int(p.stem.split("-")[-1]))]
        return outs, ms


def run_qnn(pte: Path, inputs: Sequence[np.ndarray], iterations: int = 20) -> tuple[list[np.ndarray], float]:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = _write_inputs(tmp, inputs)
        (tmp / "input_list.txt").write_text(" ".join(map(str, paths)) + "\n")
        (tmp / "outputs").mkdir()
        cmd = [str(QNN_RUNNER), f"--model_path={pte}", f"--input_list_path={tmp / 'input_list.txt'}",
               f"--output_folder_path={tmp / 'outputs'}", f"--iteration={iterations}", "--warm_up=3",
               f"--performance_output_path={tmp / 'perf.txt'}"]
        env = dict(os.environ, LD_LIBRARY_PATH=f"{QNN_BACKEND_LIB_DIR}:/usr/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
                   ADSP_LIBRARY_PATH="/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp;/dsp")
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"qnn_executor_runner failed ({res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
        # perf.txt holds the total time of all --iteration runs for the single input line, not the mean.
        total = float((tmp / "perf.txt").read_text().strip()) if (tmp / "perf.txt").exists() else float("nan")
        ms = total / iterations
        outs = [np.fromfile(p, dtype=np.float32) for p in
                sorted((tmp / "outputs").glob("output_0_*.raw"), key=lambda p: int(p.stem.split("_")[-1]))]
        return outs, ms


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    ref, got = ref.reshape(-1).astype(np.float64), got.reshape(-1).astype(np.float64)
    if ref.size != got.size:
        return {"error": f"size mismatch {ref.size} vs {got.size}"}
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    return {"max_abs": float(np.abs(ref - got).max()), "mean_abs": float(np.abs(ref - got).mean()), "cosine": cos}
