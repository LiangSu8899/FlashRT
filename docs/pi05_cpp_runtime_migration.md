# PI0.5 C++ runtime migration notes

This note covers externally visible behavior of the native PI0.5 producer. It
does not add a model-specific ABI: consumers continue to discover and drive
the generic `frt_model_runtime_v1` interface.

## Image input

The model-runtime `IMAGE/STAGED` port accepts explicit `RGB8`, `u8`, HWC host
images. It rejects BGR, grayscale, unsupported layouts, invalid dimensions,
and short strides instead of guessing a conversion.

The legacy `frt_pi05_runtime_prepare_vision` entry remains source-compatible
with its explicit RGB, BGR, RGBA, BGRA, and grayscale formats. Existing OpenCV
BGR callers can remain on that entry or convert to RGB before using the generic
model-runtime face.

Pixel normalization follows the reference float32 operation order
`value / 127.5 - 1`. Replacing division with a precomputed reciprocal can alter
FP8 quantization boundaries and is not equivalent for this producer.

## Action output

The logical `actions` STAGED output is F32 and includes the producer's declared
postprocessing. `actions_raw` is the BF16 SWAP alias for consumers that need
the model-space result. Consumers must select the declared port rather than
infer dtype or normalization from a model name.

## Runtime adoption

A published runtime with STAGED input or output ports must install the matching
`set_input` or `get_output` verb. Declaration-only native handoff objects are
internal overlay inputs and are marked as such; they are not independently
adoptable runtimes.

Port, stage, binding-window, stream-placement, and capsule-region changes alter
the runtime fingerprint. Capsules produced under an older fingerprint must be
regenerated; rejecting their restore is required behavior.

## Native FA2 dependency

The Python FA2 adapter and the Python-free `libflashrt_fa2_raw.so` are one
install unit. Native producers link the raw library and Python producers reach
the same symbols through the adapter. Deployment packages must install both in
the same directory so their `$ORIGIN` runtime lookup remains relocatable.
