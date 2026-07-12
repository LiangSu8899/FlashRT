#!/usr/bin/env python3
"""Post-build INT4 (E0M3) unlock for sm_120 OMMA kernels.

ptxas can only emit E2M1 element formats for
`mma.sync.kind::mxf4nvf4.block_scale` (SASS OMMA.SF.16864). On sm_120 the
element format actually lives in bits 78 (A operand) / 79 (B operand) of
the 128-bit instruction: 0 = E2M1, 1 = E0M3 (uniform INT4, codebook
-7..7). This tool flips those bits for every OMMA.SF instruction inside
device functions whose (mangled) name contains a marker substring — by
convention `int4_` (see csrc/kernels/int4_w4a4_mma_sm120.cu).

Works on:
  * standalone .cubin files
  * host ELF objects (.so / executables) with an uncompressed .nv_fatbin
    — every embedded sm_120 cubin is located by ELF scan and patched in
    place. (flash_rt builds store cubins uncompressed.)

## Verification is asymmetric — read this before trusting an exit code

An UNpatched binary disassembles normally, so the sites are locatable and
their bits are readable. A PATCHED binary does NOT: once bits 78/79 are
set, `cuobjdump`/`nvdisasm` no longer decode the instruction as OMMA.SF
(it renders as something else or drops out), so the patched sites cannot
be re-located statically. Therefore:

  * `--verify --expect e2m1` : PRE-patch gate. Confirms the sites exist and
    are still E2M1. exit 0 iff every located site is E2M1; nonzero
    otherwise (incl. "no sites found"). Use this in the build to prove you
    compiled the right thing before patching.
  * `--verify --expect int4`  : on an unpatched binary this reports the
    sites as E2M1 and exits 1 (NOT patched). On a patched binary the sites
    are unlocatable and it exits 2 (INCONCLUSIVE) — static analysis cannot
    confirm a patched binary. It never returns 0 by guessing.
  * `--verify` (no --expect) : report only, always exit 0.

The AUTHORITATIVE check that a binary decodes E0M3 is the runtime canary
`int4_codebook_canary()` exported by the kernels — call it at module load
(fail-fast). Do not treat any static exit code as proof of a patched
binary; treat `--expect e2m1 == 0` as proof it is still UNpatched.

Patch mode fails loudly: if it flips zero instructions (already patched /
no marker match) it exits nonzero so a build step cannot silently ship an
unpatched or double-run binary.

Usage:
  patch_int4_omma_sm120.py <input> [-o OUT] [--marker int4_]
                           [--operands ab|a|b]
                           [--verify [--expect e2m1|int4]]
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
    """OMMA.SF sites in marked functions, or None if not disassemblable."""
    try:
        out = subprocess.run([CUOBJDUMP, "-sass", cubin_path],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None
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
    """{section_name: file_offset} from an in-memory cubin ELF64."""
    if cubin_bytes[:4] != b"\x7fELF" or cubin_bytes[4] != 2:
        raise ValueError("not an ELF64 cubin")
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
    table end and every section/segment's (offset+size). Does not assume
    the shdr table is last (it often isn't in nvcc cubins)."""
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
        if pos + 0x40 <= len(data):
            machine, = struct.unpack_from("<H", data, pos + 0x12)
            if machine == 190:  # EM_CUDA
                try:
                    size = elf_total_size(data, pos)
                    if 0 < size <= len(data) - pos:
                        yield pos, size
                        pos += size
                        continue
                except struct.error:
                    pass
        pos += 4


def has_marker_text(cubin, marker):
    """True if the cubin has a .text.<fn> section whose name matches the
    marker — used to tell a real (patchable) cubin from a stub even when
    cuobjdump cannot disassemble it (i.e. it is already patched)."""
    try:
        secs = text_section_offsets(cubin)
    except ValueError:
        return False
    return any(n.startswith(".text.") and marker in n for n in secs)


def locate_sites(cubin, marker):
    """(disasm_ok, [(fn, foff, ioff, text)]) for one cubin (bytes).
    foff = file offset of instruction byte 9 within the cubin."""
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as tf:
        tf.write(cubin)
        tmp = tf.name
    try:
        sites = sass_sites(tmp, marker)
    finally:
        os.unlink(tmp)
    if sites is None:
        return False, []
    secs = text_section_offsets(cubin)
    out = []
    for fn, ioff, text in sites:
        sec = f".text.{fn}"
        if sec not in secs:
            raise RuntimeError(f"section {sec} missing")
        out.append((fn, secs[sec] + ioff + 9, ioff, text))
    return True, out


def bits_str(v):
    return ("A" if v & BIT_A else "-") + ("B" if v & BIT_B else "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="default: patch in place")
    ap.add_argument("--marker", default="int4_")
    ap.add_argument("--operands", choices=["ab", "a", "b"], default="ab")
    ap.add_argument("--verify", action="store_true",
                    help="do not modify; report/gate on current bit state")
    ap.add_argument("--expect", choices=["e2m1", "int4"],
                    help="with --verify: pass/fail gate (see module docstring)")
    args = ap.parse_args()
    if args.expect:
        args.verify = True

    mask = (BIT_A if "a" in args.operands else 0) | \
           (BIT_B if "b" in args.operands else 0)
    data = bytearray(open(args.input, "rb").read())

    machine, = struct.unpack_from("<H", data, 0x12)
    if machine == 190:
        cubins = [(0, len(data))]
    else:
        cubins = list(find_embedded_cubins(bytes(data)))
        if not cubins:
            print("no embedded cubins found")
            sys.exit(1)

    total_patched = 0
    located = 0          # sites whose bits could be read
    matched = 0          # sites matching --expect
    mismatched = 0
    unreadable_marked = False   # a marked cubin that yields no readable sites
                                # (cuobjdump failed OR decoded no OMMA — both
                                # happen once the bits are patched)

    for off, size in cubins:
        sub = bytearray(data[off:off + size])
        ok, sites = locate_sites(sub, marker=args.marker)
        if not sites:
            # No readable OMMA. If the cubin still carries marked .text
            # sections, the instructions are there but unreadable — i.e.
            # already patched — not a marker-less/stub cubin.
            if has_marker_text(sub, args.marker):
                unreadable_marked = True
            continue
        if len(cubins) > 1:
            print(f"[cubin @0x{off:x} size 0x{size:x}]")
        for fn, foff, ioff, text in sites:
            cur = sub[foff]
            located += 1
            if args.verify:
                print(f"  {fn}+0x{ioff:x}: bits[{bits_str(cur)}]")
                if args.expect:
                    want = mask if args.expect == "int4" else 0
                    if (cur & mask) == want:
                        matched += 1
                    else:
                        mismatched += 1
            else:
                sub[foff] = cur | mask
                total_patched += 1
        if not args.verify:
            data[off:off + size] = sub

    # ── Verify mode ──
    if args.verify:
        if not args.expect:
            print(f"report only ({located} site(s)); "
                  f"use --expect e2m1|int4 for a pass/fail gate")
            sys.exit(0)
        if mismatched:
            print(f"FAIL: {mismatched}/{located} site(s) are not "
                  f"'{args.expect}'")
            sys.exit(1)
        if args.expect == "int4" and unreadable_marked:
            print("INCONCLUSIVE: a marked cubin could not be disassembled "
                  "— expected once patched (cuobjdump cannot read patched "
                  "OMMA). Static analysis cannot confirm a patched binary; "
                  "run int4_codebook_canary() at load for the authoritative "
                  "check.")
            sys.exit(2)
        if located == 0:
            if unreadable_marked:
                print("INCONCLUSIVE: marked cubin present but not "
                      "disassemblable; use the runtime canary.")
                sys.exit(2)
            print(f"FAIL: no OMMA.SF sites match marker '{args.marker}'")
            sys.exit(1)
        print(f"OK: all {located} site(s) are '{args.expect}'")
        sys.exit(0)

    # ── Patch mode ──
    if total_patched == 0:
        if unreadable_marked:
            print("nothing patched: a marked cubin could not be "
                  "disassembled — the binary may already be patched "
                  "(cuobjdump cannot read patched OMMA). Rebuild to re-patch "
                  "from source.")
            sys.exit(3)
        print(f"nothing patched: no OMMA.SF sites match marker "
              f"'{args.marker}'")
        sys.exit(1)
    out = args.output or args.input
    open(out, "wb").write(data)
    print(f"patched {total_patched} OMMA.SF instruction(s) "
          f"(operands={args.operands}) -> {out}")


if __name__ == "__main__":
    main()
