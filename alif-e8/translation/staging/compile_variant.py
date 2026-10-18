"""Recompile a frozen quantized partition with an alternate Vela memory mode."""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export as full_export
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.common.pipeline_config import ArmPassPipelineConfig, SoftmaxDecompositionConfig
from prepare import ARTIFACTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--memory-mode", default="Shared_Sram")
    parser.add_argument("--extra-flag", action="append", default=[])
    args = parser.parse_args()
    destination = ARTIFACTS / "staging/variants"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source, destination / (args.name+"_a16w8.pt2"))
    full_export.ARTIFACTS = destination

    def specification():
        spec = EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SRAM_OSPI",
                                memory_mode=args.memory_mode,
                                config_ini=str(Path(__file__).resolve().parents[2] / ".cmsis/ensemble_vela.ini"),
                                extra_flags=["--verbose-performance", "--verbose-allocation", *args.extra_flag])
        spec.set_pass_pipeline_config(ArmPassPipelineConfig(softmax=SoftmaxDecompositionConfig.STABLE))
        return spec

    full_export.compile_spec = specification
    full_export.compile_model(args.name)


if __name__ == "__main__":
    main()
