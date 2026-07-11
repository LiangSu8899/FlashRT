# FA2 native C library

FlashRT builds vendored FA2 entry points into a Python-free shared library:

- `libflashrt_fa2_raw.so` owns the `fvk_attention_fa2_fwd_*` C symbols;
- `flash_rt_fa2` is the Python adapter and links the raw library;
- native C++ runtimes link the same raw library directly.

This keeps one implementation and one symbol set for Python and native
producers. Model-specific code must not compile a private copy of the FA2
wrappers.

This is an internal native linkage boundary, not a new versioned public ABI.
Consumers outside the FlashRT build must continue to use the supported
runtime interfaces rather than bind these kernel entry points directly.

## Packaging contract

The Python adapter and raw library are one install unit. Deployments that use
`flash_rt_fa2` must install both files in the same directory. Both targets use
an `$ORIGIN` runtime search path, so no build-tree path is part of deployment.

Copying only the Python extension is no longer supported. Packaging and image
rules should select the CMake install targets instead of copying an individual
extension file.

## C boundary

The raw library has no Python dependency. Its declarations live in
`csrc/attention/fa2_wrapper.h`; the Python adapter includes that header instead
of maintaining a second set of declarations. Existing `fvk_*` signatures are
unchanged.

The causal entry is always present when FA2 is enabled. Unsupported dtype/head
dimension combinations fail explicitly according to the compiled FA2 matrix;
adding a model consumer must not silently change that matrix.

## Validation

- the raw library exports the expected `fvk_attention_fa2_fwd_*` symbols;
- it has no unresolved Python or `fvk_*` symbols;
- the Python adapter has a dynamic dependency on the raw library;
- adapter and raw-library runtime search paths are `$ORIGIN`;
- the install smoke test rejects an adapter without its raw library;
- the Python adapter and native C++ attention gates both execute against the
  shared implementation.
