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
postprocessing. `actions_raw` is the producer-declared SWAP alias for consumers
that need the model-space result: BF16 on SM120 and F16 on SM110. Consumers
must select the declared port rather than infer dtype or normalization from a
model name.

## Native precision routes

The native producer supports SM120 BF16, SM120 static FP8 E4M3, and SM110
static FP8 E4M3. Every FP8 route requires a compatible calibration artifact;
SM120 uses the v2 artifact with vision, encoder, and decoder scales, while
SM110 uses the v1 artifact with encoder and decoder scales. `precision="auto"`
selects SM120 FP8 only when `calibration_path` is present, otherwise SM120 BF16;
SM110 auto-selects FP8 and therefore still requires the artifact.

Public BF16 windows on SM120 describe staging and attention-boundary storage,
not the GEMM precision. An FP8 claim must be verified from producer identity,
artifact metadata, and captured kernel dispatch. Native C++ NVFP4 is not
implemented; Python precision routes remain independent.

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
