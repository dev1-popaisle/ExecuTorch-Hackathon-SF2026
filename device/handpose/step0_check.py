"""Step 0: confirm the on-device toolchain end to end (export -> .pte -> runner) for XNNPACK and QNN HTP."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from export_lib import QNN_RUNNER, SOC, XNN_RUNNER, compare, delegation_summary, export_qnn, export_xnnpack, run_qnn, run_xnnpack

OUT = Path(__file__).resolve().parent.parent / "export" / "step0"


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(32, 10))

    def forward(self, x):
        return (self.net(x),)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "kernel": platform.release(),
        "soc": Path("/sys/devices/soc0/machine").read_text().strip(),
        "htp_arch": "V75" if Path("/usr/lib/dsp/cdsp/libQnnHtpV75Skel.so").exists() else "unknown",
        "torch": torch.__version__,
        "runners": {"xnnpack": XNN_RUNNER.exists(), "qnn": QNN_RUNNER.exists()},
    }
    qairt = subprocess.run(["dpkg-query", "-W", "-f=${Package} ${Version}\\n", "qairt-*"], capture_output=True, text=True)
    report["qairt_packages"] = qairt.stdout.split("\n")[:-1]

    torch.manual_seed(0)
    model = Tiny().eval()
    x = torch.rand(1, 3, 64, 64)
    ref = model(x)[0].detach().numpy()

    xnn = export_xnnpack(model, (x,), OUT / "tiny_xnnpack.pte")
    outs, ms = run_xnnpack(xnn, [x.numpy()])
    report["xnnpack"] = {**compare(ref, outs[0]), "ms": ms}

    calib = [torch.rand(1, 3, 64, 64) for _ in range(16)] + [x]
    qnn = export_qnn(model, (x,), calib, OUT / "tiny_qnn_8a8w.pte", quant="8a8w")
    report["qnn_delegation"] = delegation_summary(qnn)
    outs, ms = run_qnn(qnn, [x.numpy()])
    report["qnn"] = {**compare(ref, outs[0]), "ms": ms, "soc": SOC}

    print(json.dumps(report, indent=2))
    ok = report["xnnpack"]["cosine"] > 0.999 and report["qnn"]["cosine"] > 0.98 and report["qnn_delegation"]["delegates"]
    (OUT / "step0_report.json").write_text(json.dumps(report, indent=2))
    print("STEP0_OK" if ok else "STEP0_FAIL")


if __name__ == "__main__":
    main()
