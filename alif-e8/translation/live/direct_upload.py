#!/usr/bin/env python3
"""Run load_assets through one persistent J-Link SDK connection and 32-bit writes."""
import ctypes
import os
import shlex
import sys
import time
from pathlib import Path

import load_assets


class DirectConnection:
    def __init__(self):
        self.link = None

    def connect(self, args):
        if self.link is not None:
            return
        sys.path.insert(0, os.environ.get("DCOPS_PYLINK_PATH", "/tmp/dcops-pylink"))
        import pylink
        library = pylink.library.Library(dllpath=os.environ.get(
            "DCOPS_JLINK_LIBRARY", "/Applications/SEGGER/JLink_V986a/libjlinkarm.9.86.1.dylib"))
        self.link = pylink.JLink(lib=library)
        self.link.open()
        self.link.set_tif(pylink.enums.JLinkInterfaces.SWD)
        self.link.connect("AE822FA0E5597LS0_M55_HP", speed=args.speed)
        self.link.exec_command("SetEnableFlashbank 0xA0000000=0")
        self.link.exec_command("SetAllowFlashCache=0")
        self.write = self.link._dll.JLINKARM_WriteMemEx
        self.write.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        self.write.restype = ctypes.c_int
        print(f"Persistent J-Link SDK connection ready, SWD{args.speed}kHz; target remains running", flush=True)

    def close(self):
        if self.link is not None:
            self.link.close()
            self.link = None

    def write_bytes(self, address, raw, deadline):
        for offset in range(0, len(raw), 64*1024):
            if time.monotonic() >= deadline:
                raise RuntimeError("Direct J-Link chunk write exceeded its deadline")
            block = raw[offset:offset+64*1024]
            aligned = len(block)&~3
            for start, count, width in ((0, aligned, 4), (aligned, len(block)-aligned, 1)):
                if not count:
                    continue
                buffer = ctypes.create_string_buffer(block[start:start+count])
                written = self.write(address+offset+start, count, buffer, width)
                if written != count:
                    raise RuntimeError(f"JLINKARM_WriteMemEx at0x{address+offset+start:08x} wrote{written}/{count} bytes")
            print(f"  SDK write0x{address+offset:08x}: {offset+len(block)}/{len(raw)} bytes", flush=True)
        if time.monotonic() >= deadline:
            raise RuntimeError("Direct J-Link chunk write exceeded its deadline")

    def commands(self, args, directory, name, commands, timeout=None):
        self.connect(args)
        deadline = time.monotonic()+(args.timeout if timeout is None else timeout)
        log = []
        try:
            for command in commands:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Direct J-Link {name} exceeded its deadline")
                parts = shlex.split(command)
                log.append(command)
                if parts[0] == "loadfile":
                    if len(parts) != 4 or parts[3] != "noreset":
                        raise RuntimeError("Direct upload accepts only noreset RAM loads")
                    self.write_bytes(int(parts[2], 0), Path(parts[1]).read_bytes(), deadline)
                elif parts[0] == "w4":
                    address, value = int(parts[1], 0), int(parts[2], 0)
                    if self.link.memory_write32(address, [value]) != 4:
                        raise RuntimeError(f"Flag write failed at0x{address:08x}")
                elif parts[0] == "savebin":
                    path, address, count = Path(parts[1]), int(parts[2], 0), int(parts[3], 0)
                    if count == 4 and address%4 == 0:
                        values = self.link.memory_read32(address, 1)
                        if len(values) != 1:
                            raise RuntimeError(f"Flag read failed at0x{address:08x}")
                        raw = int(values[0]).to_bytes(4, "little")
                    else:
                        raw = bytes(self.link.memory_read8(address, count))
                    if len(raw) != count:
                        raise RuntimeError(f"Readback at0x{address:08x} returned{len(raw)}/{count} bytes")
                    path.write_bytes(raw)
                else:
                    raise RuntimeError(f"Unsupported direct API command: {command}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Direct J-Link {name} exceeded its deadline")
        except Exception as error:
            log.append("ERROR: "+str(error))
            raise
        finally:
            (directory / (name+".direct.log")).write_text("\n".join(log)+"\n")


def main():
    connection = DirectConnection()
    load_assets.jlink = connection.commands
    try:
        load_assets.main()
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Direct asset upload failed: {error}", file=sys.stderr)
        raise SystemExit(1)
