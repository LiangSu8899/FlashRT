# Jetson-PI Validation

This matrix defines the merge gates for the optional provider. Commands use
placeholders intentionally; validation records must not publish workstation
paths, device assignments, model storage locations, or private environment
details.

## Required Gates

| Area | Gate |
|---|---|
| Default isolation | Provider options OFF builds no provider target or DSO and changes no existing model route. |
| Provider-only CI | Contract tests build with CUDA, exec, and Jetson-PI disabled. |
| Public ABI | Provider returns `frt_model_runtime_v1`; no parallel runtime ABI is defined. |
| Extension | Exactly one valid `GENERIC_STAGE_PLAN_V1` authority is published. |
| DSO surface | Dynamic exports equal `frt_model_runtime_open_v1` plus the version node. |
| Lifecycle | Open, retain/release, repeated instances, and all failure paths are leak-free. |
| Identity | Checkpoint bytes, backend, port schema, and selected plan affect the fingerprint. |
| VLA | Whole-step `infer`, split `context -> action`, and the direct provider path produce identical action buffers. The split is published only when the backend declares a real context/action capability. |
| LLM | Whole generation and host-driven `reset -> prefill -> decode` match exactly under deterministic sampling. |
| MLLM | Whole generation and staged decode match exactly under deterministic sampling. |
| Install tree | An out-of-tree consumer loads the installed DSO without source-tree headers or build-tree RPATH. |
| Nexus | Generic OPAQUE execution is synchronous, dependency ordered, and does not alter native graph adoption. |

## Contract Tests

The repository keeps focused tests for:

- old-prefix and current engine-vtable compatibility;
- strict JSON parsing and checkpoint identity;
- port schema, selected-plan records, stable executor references, and DAG
  dependencies;
- missing staged verbs, stale output, failed replacement, decode budget, and
  multi-instance isolation;
- direct narrow-API parity for VLA, LLM, and MLLM paths;
- Python whole-request and staged APIs;
- exact provider DSO exports.

Tests requiring model files skip explicitly when fixtures are unavailable.
Skipping is not parity evidence; release qualification must provide the model
matrix below.

## Release Matrix

At least one supported device for each advertised backend must run:

| Model face | Whole request | Selected plan | Direct parity | Repeated session |
|---|---:|---:|---:|---:|
| VLA | required | `context -> action` | required | required |
| Text LLM | required | required | required | required |
| Multimodal LLM | required | required | required | required |

VLA qualification uses a backend that explicitly declares the real
context/action capability. The merged backend change
[Jetson-PI-Edge#1](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge/pull/1)
provides the PI0.5 encode/decode boundary and prompt-state serialization. The
follow-up [Jetson-PI-Edge#2](https://github.com/PKU-SEC-Lab/Jetson-PI-Edge/pull/2)
extends the real boundary to legacy Pi0 with owned VIT embeddings and publishes
the cross-model capability contract. The opt-in parity tests also require
actual GGUF model-kind detection, one-shot pending-context semantics, and
whole/split byte equality.

For deterministic paths, text/token outputs are exact and VLA action buffers
are byte-identical unless the provider documents a backend-specific numerical
contract. Shape, dtype, finite-value, stale-state, and teardown checks are
always required.

## Portable Build Check

```bash
cmake -S cpp -B build/provider-test \
  -DBUILD_TESTING=ON \
  -DFLASHRT_CPP_WITH_LLAMA_CPP_PROVIDER=ON \
  -DFLASHRT_CPP_WITH_JETSON_PI=ON \
  -DJetsonPI_ROOT=<jetson-pi-install-prefix>
cmake --build build/provider-test
ctest --test-dir build/provider-test --output-on-failure
```

After installation, audit the DSO using the platform's dynamic-symbol and
runtime-dependency tools. Linux acceptance requires:

```text
defined public symbol: frt_model_runtime_open_v1
RPATH/RUNPATH:          $ORIGIN or no embedded path
build/source paths:     absent
```

The final cross-repository gate installs FlashRT provider and Nexus into clean
prefixes, then runs both the native graph producer and this generic provider
through the same Nexus host lifecycle.
