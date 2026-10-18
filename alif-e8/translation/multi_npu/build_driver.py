"""Build architecture-dispatched U55/U85 device and PMU objects for one MCU."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PACKS = Path.home() / '.cache/arm/packs'
DRIVER = PACKS / 'ARM/ethos-u-core-driver/1.26.5/ethos_u_core_driver'
CMSIS = PACKS / 'ARM/CMSIS/6.4.0/CMSIS/Core/Include'
TOOLS = Path.home() / '.vcpkg/artifacts/2139c4c6/compilers.arm.arm.none.eabi.gcc/15.3.1/bin'


def prototypes(path, prefix):
    source = re.sub(r'/\*.*?\*/', '', path.read_text(), flags=re.S)
    return [(ret.strip(), name, ' '.join(args.split())) for ret, name, args in
            re.findall(r'\n([\w ]+?)\s+(' + prefix + r'\w+)\s*\(([^;{}]*)\)\s*;', source)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/translator/mixed-driver')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = prototypes(DRIVER / 'src/ethosu_device.h', 'ethosu_dev_')
    pmu = prototypes(DRIVER / 'include/pmu_ethosu.h', 'ETHOSU_PMU_')
    assert len(device) == 10, len(device)
    assert len(pmu) >= 24, len(pmu)
    functions = device + pmu
    wrappers = ['#include "ethosu_device.h"', '#include "ethosu_driver.h"', '#include "pmu_ethosu.h"',
                '#include <stdint.h>',
                'static int is_u85(const struct ethosu_device *dev) { return (uintptr_t)dev->reg == 0x49042000u; }']
    pmu_text = (DRIVER / 'include/pmu_ethosu.h').read_text()
    enums = re.findall(r'enum ethosu_pmu_event_type\s*\{(.*?)\}', pmu_text, re.S)
    names = [re.findall(r'ETHOSU_PMU_\w+', enum) for enum in enums]
    assert len(names) == 2
    to_u55 = [names[0].index(name) if name in names[0] else 255 for name in names[1]]
    to_u85 = [names[1].index(name) if name in names[1] else 255 for name in names[0]]
    wrappers += ['static const unsigned char to_u55[] = {' + ','.join(map(str, to_u55)) + '};',
                 'static const unsigned char to_u85[] = {' + ','.join(map(str, to_u85)) + '};',
                 'static unsigned map_event(unsigned value, const unsigned char *map, unsigned count) { return value < count ? map[value] : 255; }']
    for architecture in ('u55', 'u85'):
        (out / f'{architecture}_names.h').write_text(''.join(
            f'#define {name} {architecture}_{name}\n' for _, name, _ in functions) + f'#define ethosu_config_select {architecture}_ethosu_config_select\n')
    for ret, name, arguments in functions:
        for architecture in ('u55', 'u85'):
            wrappers.append(f'{ret} {architecture}_{name}({arguments});')
        values = [re.search(r'(\w+)\s*$', value).group(1) for value in arguments.split(',')] if arguments != 'void' else []
        call_args = ', '.join(values)
        if not values:
            # U55 has four counters, U85 has eight; expose the shared safe subset.
            assert name == 'ETHOSU_PMU_Get_NumEventCounters'
            body = 'return 4;'
        else:
            first = values[0]
            test = f'(uintptr_t){first} == 0x49042000u' if name == 'ethosu_dev_init' else (
                f'is_u85(&{first}->dev)' if name.startswith('ETHOSU_PMU_') else f'is_u85({first})')
            # init must select using base_address: dev->reg is not initialized yet.
            if name == 'ethosu_dev_init':
                test = f'(uintptr_t){values[1]} == 0x49042000u'
            statement = '' if ret == 'void' else 'return '
            u55_args = list(values)
            for index, declaration in enumerate(arguments.split(',')):
                if 'enum ethosu_pmu_event_type' in declaration:
                    u55_args[index] = f'map_event({values[index]}, to_u55, sizeof(to_u55))'
            u55_call = f'u55_{name}({", ".join(u55_args)})'
            if ret == 'enum ethosu_pmu_event_type':
                u55_call = f'map_event({u55_call}, to_u85, sizeof(to_u85))'
            body = (f'if ({test}) {{ {statement}u85_{name}({call_args}); }} '
                    f'else {{ {statement}{u55_call}; }}')
        wrappers.append(f'{ret} {name}({arguments}) {{ {body} }}')
    wrapper = out / 'mixed_device.c'
    wrapper.write_text('\n'.join(wrappers) + '\n')
    cc = TOOLS / 'arm-none-eabi-gcc'
    flags = ['-mcpu=cortex-m55', '-mthumb', '-mfloat-abi=hard', '-std=c11', '-O2', '-ffunction-sections', '-fdata-sections',
             '-Wall', '-Wextra', '-Werror', '-I' + str(DRIVER / 'include'), '-I' + str(DRIVER / 'src'), '-I' + str(CMSIS)]
    objects = []
    commands = []
    for architecture, device_source in [('u55', 'ethosu_device_u55_u65.c'), ('u85', 'ethosu_device_u85.c')]:
        for source in [DRIVER / 'src' / device_source, DRIVER / 'src/ethosu_pmu.c']:
            output = out / f'{architecture}_{source.stem}.o'
            architecture_flags = [f'-DAXI_LIMIT{index}_MEM_TYPE=3' for index in range(4)] if architecture == 'u55' else []
            command = [str(cc), *flags, *architecture_flags, '-D' + architecture.upper().replace('U', 'ETHOSU'), '-DETHOSU_ARCH=' + architecture, '-include', str(out / f'{architecture}_names.h'), '-c', str(source), '-o', str(output)]
            subprocess.run(command, check=True)
            commands.append(command)
            objects.append(output)
    for source in [DRIVER / 'src/ethosu_driver.c', wrapper, Path(__file__).with_name('dispatch.c')]:
        output = out / (source.stem + '.o')
        command = [str(cc), *flags, '-DETHOSU85', '-DETHOSU_ARCH=u85', '-c', str(source), '-o', str(output)]
        subprocess.run(command, check=True)
        commands.append(command)
        objects.append(output)
    library = out / 'libe8_ethosu_mixed.a'
    subprocess.run([str(TOOLS / 'arm-none-eabi-ar'), 'rcs', str(library), *map(str, objects)], check=True)
    # Partial link catches duplicate strong definitions and confirms dispatch resolves.
    subprocess.run([str(cc), '-nostdlib', '-r', *map(str, objects), '-o', str(out / 'mixed_combined.o')], check=True)
    (out / 'build.json').write_text(json.dumps({'library': str(library), 'device_functions': len(device), 'pmu_functions': len(pmu), 'commands': commands}, indent=2) + '\n')
    print(library)


if __name__ == '__main__':
    main()
