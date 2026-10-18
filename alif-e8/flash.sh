#!/bin/sh
set -eu
cd "$(dirname "$0")"
TASK_NM=/Users/npow/.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin/arm-none-eabi-nm
TASK_ENTRY=$("$TASK_NM" -n out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf | awk '$3 == "Reset_Handler" {print "0x" $1}')
test -n "$TASK_ENTRY"
JLinkExe -device AE822FA0E5597LS0_M55_HP -if SWD -speed 4000 -autoconnect 1 <<SCRIPT
h
loadfile $PWD/out/cmsis-executorch/DevKit-E8/Debug/cmsis-executorch.elf
r
w4 0xE000ED08 0x80200000
wreg MSP 0x20100000
SetPC $TASK_ENTRY
w4 0xE0002008 0
w4 0xE000200C 0
w4 0xE0002010 0
w4 0xE0002014 0
w4 0xE0002018 0
w4 0xE000201C 0
w4 0xE0002020 0
w4 0xE0002024 0
g
exit
SCRIPT
