# Building FlashRT on Windows

Linux remains the primary, fully tested platform (see [INSTALL.md](INSTALL.md)).
This document captures the Windows-specific build path for users who need
local Blackwell (RTX 5090, SM120) inference on a native Windows host.

If anything in this guide diverges from the Linux instructions, follow
the Linux path on Linux — the Windows fixes below are conditional and
must not affect Linux builds.

---

## 1. Prerequisites

| Component | Version | Notes |
|---|---|---|
| GPU | RTX 5090 (SM120) / RTX 4090 (SM89) | Blackwell support requires CUDA 12.8+ |
| CUDA Toolkit | 12.9 (recommended) or 13.0 | Default install path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9` |
| Visual Studio | 2022 (MSVC 19.40+) or 2026 (MSVC 19.50+) | Install the "Desktop development with C++" workload |
| CMake | 3.24+ | |
| Ninja | latest | `pip install ninja` or VS-bundled |
| Python | 3.10 / 3.11 / 3.12 | Same Python must run cmake AND `import flash_rt` |
| pybind11 | latest | `pip install pybind11` |
| CUTLASS | 4.4.2 | Manual clone — see step 3 |

---

## 2. Open the right developer shell

Open **"x64 Native Tools Command Prompt for VS 2022"** (or 2026). This
sets `cl.exe`, `link.exe`, the Windows SDK, and the MSVC runtime on
`PATH`. A plain `cmd` / `PowerShell` will not find the host compiler.

Inside that shell, activate your Python venv:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -U pip pybind11 ninja torch
```

---

## 3. Clone CUTLASS

```bat
git clone --depth 1 --branch v4.4.2 ^
    https://github.com/NVIDIA/cutlass.git third_party\cutlass
```

(Same as Linux — CUTLASS is intentionally not vendored.)

---

## 4. Configure & build

```bat
mkdir build
cd build
cmake -G Ninja ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DGPU_ARCH=120 ^
      ..
ninja -j16
```

`GPU_ARCH=120` targets the RTX 5090; use `89` for 4090. Auto-detection
via `nvidia-smi` works on Windows too if you omit the flag.

When the build reaches `[Linking CXX shared module
flash_rt_kernels.cp311-win_amd64.pyd]`, you're done.

For an editable install (recommended, mirrors Linux):

```bat
cd ..
pip install -e ".[torch]"
```

The `.pyd` files are dropped into `flash_rt\` automatically.

---

## 5. Why three Windows-specific fixes exist

Three classes of Linux-only assumptions used to break Windows builds.
The repo now handles all three; this section documents what each
one does, so the workarounds aren't reinvented as one-off patches.

### 5.1 Cross-platform dynamic library loading
`csrc/attention/fmha_dispatch.cu` originally included `<dlfcn.h>` and
called `dlopen` / `dlsym` / `dlclose` directly. POSIX-only, so MSVC
fails at the header. The file now ships a tiny `dyn_open` /
`dyn_sym` / `dyn_close` wrapper that compiles to `LoadLibraryA` /
`GetProcAddress` / `FreeLibrary` under `_WIN32` and to the original
POSIX calls everywhere else. Behavior on Linux is unchanged byte-for-byte.

If you ship a `.dll` build of the SM100/SM110 CUTLASS FMHA kernel,
pass its full path to `load_fmha_library("...\\fmha_fp16_strided.dll")`.
The default `flash_rt_kernels` build does not produce this DLL on
SM120 (FA2 in-process is faster on Blackwell), so most Windows users
never load it.

### 5.2 Explicit-instantiation guard for MSVC
The kernel `.cu` files (norm, activation, quantize, elementwise, rope,
fusion) end with `template __global__ void f<__half>(const __half*, ...);`
lines. MSVC 19.50 rejects these explicit instantiations when the
signature combines `const`-pointer parameters with typedef aliases
like `__half` — GCC and Clang accept them.

The fix in `csrc/kernels/common.cuh`:

```cpp
#ifdef _MSC_VER
  #define FVK_KERNEL_INSTANTIATE(decl) /* implicit on MSVC */
#else
  #define FVK_KERNEL_INSTANTIATE(decl) template decl;
#endif
```

Each `template __global__ void X<T>(...);` was rewritten as
`FVK_KERNEL_INSTANTIATE(__global__ void X<T>(...))`. On Linux nothing
changes — the macro expands back to the original explicit
instantiation. On MSVC the macro disappears, and nvcc/MSVC implicitly
instantiate the kernel from the host `kernel<<<...>>>` launch in the
same `.cu` file.

**Do not delete the `FVK_KERNEL_INSTANTIATE(...)` lines.** If a
future change adds a cross-TU caller of these templates, the explicit
Linux instantiations are what makes the link succeed; the MSVC branch
will need a different remedy at that point.

### 5.3 Windows DLL search-path injection
Python 3.8+ on Windows ignores `PATH` when resolving DLL dependencies
of C extensions (a deliberate security hardening). Without
intervention, `import flash_rt` fails with
`ImportError: DLL load failed while importing flash_rt_kernels`,
because `cudart64_*.dll`, `cublas64_*.dll`, etc. live in the CUDA
toolkit `bin\` directory which the secure loader does not consult.

`flash_rt/__init__.py` calls `os.add_dll_directory()` for every
plausible CUDA install (the active `CUDA_PATH`, the default v13.0 /
v12.9 / v12.8 install paths, and `CUDNN_PATH` if set), gated on
`sys.platform == 'win32'` so Linux is untouched. If your CUDA install
is somewhere else, set `CUDA_PATH` in the environment before
importing `flash_rt`.

### 5.4 MSVC compile flags in `CMakeLists.txt`
The top of `CMakeLists.txt` adds an `if(MSVC) ... endif()` block that
attaches `/bigobj` (FA2 templates blow past the default section
limit), `/utf-8` (sources contain non-ASCII comments), `/Zc:__cplusplus`
(make CUTLASS feature detection see the right C++ standard) and
`/EHsc` (pybind11) to both CXX and CUDA-host compile lines. The FA2
target's `-Xcompiler -Wno-deprecated-declarations` is mapped to
`/wd4996` under MSVC; everywhere else it stays as the GCC form.

---

## 6. Smoke test

```bat
python -c "import flash_rt; print(flash_rt.__version__)"
python -c "from flash_rt import flash_rt_kernels; print('kernels OK')"
```

Both commands should print without error. If the second one fails
with `ImportError: DLL load failed`, recheck section 5.3 — usually
your `CUDA_PATH` env var is unset and your toolkit is in a non-default
location.

---

## 7. Known gaps vs. Linux

- **`fmha_fp16_strided.dll`** is not currently part of the default
  Windows build target; SM120 (RTX 5090) routes attention through
  the in-process FA2 path, which is what the leaderboard numbers
  come from. SM100/SM110 users who want the CUTLASS FMHA backend on
  Windows will need to add a Windows-side build target — open an
  issue if you need this.
- **JAX FFI module** (`flash_rt_jax_ffi`) is detected only when
  `jax.ffi.include_dir()` resolves; JAX on Windows is not officially
  supported by upstream JAX, so this module is normally skipped.
- **Editable install + Ninja** is the tested combination. Other
  generators (Visual Studio MSBuild, Make) may work but are not
  exercised in CI.
