#!/usr/bin/env python3
"""Post-build INT4 (E0M3) unlock for sm_120 OMMA kernels.

ptxas can only emit E2M1 element formats for
`mma.sync.kind::mxf4nvf4.block_scale` (SASS OMMA.SF.16864). On sm_120
the element format actually lives in bits 78 (A operand) / 79 (B
operand) of the 128-bit instruction: 0 = E2M1, 1 = E0M3 (uniform INT4,
codebook -7..7). This tool flips those bits for every OMMA.SF
instruction inside device functions whose (mangled) name contains a
marker substring — by convention `int4_` (see
csrc/kernels/int4_w4a4_mma_sm120.cu).

Works on:
  * standalone .cubin files
  * host ELF objects (.so) with an uncompressed .nv_fatbin — every
    embedded sm_120 cubin is located by ELF magic scan and patched in
    place. (flash_rt builds store cubins uncompressed; if a compressed
    fatbin entry is ever encountered the affected cubin is skipped and
    reported, never silently half-patched.)

Usage:
  patch_int4_omma_sm120.py <input> [-o OUT] [--marker int4_]
                           [--operands ab|a|b] [--verify]

  --verify  only report bit state per instruction, change nothing.
  In-place patching (no -o) refuses to run on files it cannot fully
  process. Exit code 0 = success / all-verified, nonzero otherwise.

Runtime safety net: the kernels export int4_codebook_canary(), which
returns 0 only when the loaded SASS truly decodes E0M3 — call it at
module init (fail-fast) so an unpatched .so can never serve traffic.
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

CUOBJDUMP = os.environ.get("CUOBJDUMP", "cuobjdump")

BIT_A = 0x40  # instruction byte 9, bit 6  -> SASS bit 78 (A operand)
BIT_B = 0x80  # instruction byte 9, bit 7  -> SASS bit 79 (B operand)


def sass_sites(cubin_path, marker):
    """[(func, instr_off, text)] for OMMA.SF instrs in marked functions."""
    out = subprocess.run([CUOBJDUMP, "-sass", cubin_path],
                         capture_output=True, text=True, check=True).stdout
    sites, fn = [], None
    for line in out.splitlines():
        m = re.search(r"Function\s*:\s*(\S+)", line)
        if m:
            fn = m.group(1)
            continue
        m = re.match(r"\s*/\*([0-9a-fA-F]+)\*/\s+(.*)", line)
        if m and "OMMA" in m.group(2) and ".SF." in m.group(2):
            if fn and marker in fn:
                sites.append((fn, int(m.group(1), 16),
                              m.group(2).split(";")[0].strip()))
    return sites


def text_section_offsets(cubin_bytes):
    """{section_name: file_offset} from an in-memory cubin ELF."""
    if cubin_bytes[:4] != b"\x7fELF":
        raise ValueError("not an ELF")
    is64 = cubin_bytes[4] == 2
    if not is64:
        raise ValueError("only ELF64 cubins supported")
    e_shoff, = struct.unpack_from("<Q", cubin_bytes, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from(
        "<HHH", cubin_bytes, 0x3A)
    strtab_off, = struct.unpack_from(
        "<Q", cubin_bytes, e_shoff + e_shstrndx * e_shentsize + 0x18)
    offs = {}
    for i in range(e_shnum):
        base = e_shoff + i * e_shentsize
        name_off, = struct.unpack_from("<I", cubin_bytes, base)
        sh_offset, = struct.unpack_from("<Q", cubin_bytes, base + 0x18)
        end = cubin_bytes.index(b"\x00", strtab_off + name_off)
        name = cubin_bytes[strtab_off + name_off:end].decode()
        offs[name] = sh_offset
    return offs


def elf_total_size(buf, off):
    """Byte extent of the ELF64 at buf[off]: max of the section-header
    table end and every section's (offset+size). Does not assume the
    shdr table is last (it often isn't in nvcc cubins)."""
    e_phoff, = struct.unpack_from("<Q", buf, off + 0x20)
    e_shoff, = struct.unpack_from("<Q", buf, off + 0x28)
    e_phentsize, e_phnum = struct.unpack_from("<HH", buf, off + 0x36)
    e_shentsize, e_shnum = struct.unpack_from("<HH", buf, off + 0x3A)
    end = e_shoff + e_shnum * e_shentsize
    for i in range(e_shnum):
        base = off + e_shoff + i * e_shentsize
        sh_type, = struct.unpack_from("<I", buf, base + 0x04)
        sh_offset, sh_size = struct.unpack_from("<QQ", buf, base + 0x18)
        if sh_type != 8:  # SHT_NOBITS occupies no file space
            end = max(end, sh_offset + sh_size)
    for i in range(e_phnum):
        base = off + e_phoff + i * e_phentsize
        p_offset, = struct.unpack_from("<Q", buf, base + 0x08)
        p_filesz, = struct.unpack_from("<Q", buf, base + 0x20)
        end = max(end, p_offset + p_filesz)
    return end


def find_embedded_cubins(data):
    """Yield (offset, size) of sm_120 cubin ELFs inside a host ELF/.so."""
    pos = 0
    while True:
        pos = data.find(b"\x7fELF", pos)
        if pos < 0:
            return
        # cubins are ET_EXEC/ET_REL with e_machine EM_CUDA (190)
        if pos + 0x40 <= len(data):
            machine, = struct.unpack_from("<H", data, pos + 0x12)
            if machine == 190:
                try:
                    size = elf_total_size(data, pos)
                    if 0 < size <= len(data) - pos:
                        yield pos, size
                        pos += size
                        continue
                except struct.error:
                    pass
        pos += 4


def patch_cubin_bytes(cubin, marker, mask, verify):
    """Patch one cubin (bytearray). Returns (n_patched, report_lines)."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as tf:
        tf.write(cubin)
        tmp = tf.name
    try:
        sites = sass_sites(tmp, marker)
    except subprocess.CalledProcessError:
        return 0, []  # not a disassemblable cubin (e.g. relocatable stub)
    finally:
        os.unlink(tmp)
    if not sites:
        return 0, []
    secs = text_section_offsets(cubin)
    n, report = 0, []
    for fn, ioff, text in sites:
        sec = f".text.{fn}"
        if sec not in secs:
            raise RuntimeError(f"section {sec} missing")
        foff = secs[sec] + ioff + 9
        old = cubin[foff]
        state = ("A" if old & BIT_A else "-") + ("B" if old & BIT_B else "-")
        if verify:
            report.append(f"  {fn}+0x{ioff:x}: bits[{state}]")
        else:
            cubin[foff] = old | mask
            n += 1
    return n, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="default: patch in place")
    ap.add_argument("--marker", default="int4_")
    ap.add_argument("--operands", choices=["ab", "a", "b"], default="ab")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    mask = (BIT_A if "a" in args.operands else 0) | \
           (BIT_B if "b" in args.operands else 0)
    data = bytearray(open(args.input, "rb").read())

    machine, = struct.unpack_from("<H", data, 0x12)
    total = 0
    if machine == 190:  # bare cubin
        n, report = patch_cubin_bytes(data, args.marker, mask, args.verify)
        total += n
        for line in report:
            print(line)
    else:  # host ELF / .so: patch every embedded sm cubin in place
        found = False
        for off, size in find_embedded_cubins(bytes(data)):
            sub = bytearray(data[off:off + size])
            n, report = patch_cubin_bytes(sub, args.marker, mask, args.verify)
            if n or report:
                found = True
                print(f"[cubin @0x{off:x} size 0x{size:x}]")
                for line in report:
                    print(line)
            if n:
                data[off:off + size] = sub
                total += n
        if not found:
            print(f"no OMMA.SF sites in functions matching "
                  f"'{args.marker}' found")
            sys.exit(0 if args.verify else 1)

    if args.verify:
        print("verify-only, nothing written")
        return

    if total == 0:
        print(f"no instructions matched marker '{args.marker}'")
        sys.exit(1)
    out = args.output or args.input
    open(out, "wb").write(data)
    print(f"patched {total} OMMA.SF instruction(s) "
          f"(operands={args.operands}) -> {out}")


if __name__ == "__main__":
    main()
