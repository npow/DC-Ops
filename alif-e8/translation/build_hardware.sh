#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 translation/prepare_board.py
TASK_TOOLS=/Users/npow/.vcpkg/artifacts/2139c4c6
export PATH="$TASK_TOOLS/tools.open.cmsis.pack.cmsis.toolbox/2.15.0/bin:$TASK_TOOLS/tools.kitware.cmake/4.3.3/bin:$TASK_TOOLS/tools.ninja.build.ninja/1.13.2:$TASK_TOOLS/compilers.arm.arm.none.eabi.gcc/15.3.1/bin:$PATH"
export GCC_TOOLCHAIN_15_3_1="$TASK_TOOLS/compilers.arm.arm.none.eabi.gcc/15.3.1/bin"
export CMSIS_COMPILER_ROOT="$TASK_TOOLS/tools.open.cmsis.pack.cmsis.toolbox/2.15.0/etc"
cbuild translator.csolution.yml --context translator.Debug+DevKit-E8 "$@"
