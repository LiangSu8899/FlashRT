#!/usr/bin/env python3
"""Run the Cosmos3-Edge official Thor baseline through FlashRT dispatch."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flash_rt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Local Cosmos3-Edge directory or registered name")
    parser.add_argument("--input-json", required=True,
                        help="Official Cosmos Framework sample JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vae-path",
                        help="Local Wan2.2_VAE.pth; avoids HF download inside the container")
    parser.add_argument("--cosmos-root",
                        help="Path to a Cosmos Framework checkout")
    parser.add_argument("--config-file",
                        help="Cosmos3-Edge YAML; defaults under --cosmos-root")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--live-dump-out",
        help="Optional safetensors path for an official live denoise dump; requires official_action_only",
    )
    parser.add_argument(
        "--live-flashrt-handoff",
        action="store_true",
        help="Use official step 0 to capture live boundary, then replace later denoise velocities with FlashRT",
    )
    parser.add_argument(
        "--live-prelayer-bootstrap",
        action="store_true",
        help="Build the live FlashRT boundary before official decoder layers and skip the official step-0 transformer",
    )
    parser.add_argument(
        "--live-warm-request",
        action="store_true",
        help="Run the pre-layer FlashRT handoff with one official warmup request before the measured batch",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of official warmup requests to run before the measured request",
    )
    parser.add_argument(
        "--cache-warmup-vae",
        action="store_true",
        help="Reuse VAE latents encoded during official warmup for the measured request",
    )
    parser.add_argument(
        "--cache-warmup-prepare",
        action="store_true",
        help="Reuse prepared inference state from official warmup for the measured request",
    )
    parser.add_argument(
        "--live-boundary-out",
        help="Optional safetensors path for the step-0 live boundary used by --live-flashrt-handoff",
    )
    parser.add_argument(
        "--live-boundary-in",
        help="Optional safetensors pre-layer boundary whose tensors initialize the live FlashRT denoise engine",
    )
    parser.add_argument(
        "--live-boundary-prepare-in",
        help="Optional torch.save slim prepare artifact used to derive the live FlashRT pre-layer boundary",
    )
    parser.add_argument(
        "--live-boundary-prepare-live",
        action="store_true",
        help="Derive the live FlashRT pre-layer boundary from this request's in-memory prepared state",
    )
    parser.add_argument(
        "--live-handoff-trace-out",
        help="Optional JSON path for per-stage live FlashRT handoff timings",
    )
    parser.add_argument(
        "--upstream-trace-out",
        help="Optional JSON path for opt-in official upstream stage timings",
    )
    parser.add_argument(
        "--vae-encode-dump-out",
        help="Optional safetensors path for the measured request's VAE encode latent boundary",
    )
    parser.add_argument(
        "--vae-latent-in",
        help="Optional safetensors path whose vae_encode/output tensor replaces measured VAE encode",
    )
    parser.add_argument(
        "--vae-encode-dump-input",
        action="store_true",
        help="Also include the full VAE input video tensor in --vae-encode-dump-out",
    )
    parser.add_argument(
        "--vae-encode-profile-out",
        help="Optional JSON path for Wan2.2 VAE encode module/shape/timing profile",
    )
    parser.add_argument(
        "--vae-native-rms-silu",
        action="store_true",
        help="Enable FlashRT native BF16 RMS+SiLU inside Wan2.2 VAE encode ResidualBlocks",
    )
    parser.add_argument(
        "--vae-t1-conv2d",
        action="store_true",
        help="Run no-cache single-frame Wan2.2 CausalConv3d sites as equivalent Conv2d",
    )
    parser.add_argument(
        "--vae-native-avgdown3d",
        action="store_true",
        help="Enable FlashRT native BF16 AvgDown3D shortcut pooling inside Wan2.2 VAE encode",
    )
    parser.add_argument(
        "--vae-channels-last3d-conv320",
        action="store_true",
        help="Run steady 320-channel Wan2.2 CausalConv3d sites with channels_last_3d cuDNN layout",
    )
    parser.add_argument(
        "--vae-compile-encode",
        action="store_true",
        help="Enable upstream Wan2.2 VAE chunk AOT compile_encode for 480p 16:9",
    )
    parser.add_argument(
        "--vae-compile-trace-out",
        help="Optional JSON path for VAE compile_encode setup trace",
    )
    parser.add_argument(
        "--prepare-dump-out",
        help="Optional torch.save path for the measured request's prepared inference state",
    )
    parser.add_argument(
        "--prepare-replay-in",
        help="Optional torch.save path whose prepared inference state replaces measured prepare",
    )
    parser.add_argument(
        "--prepare-inventory-out",
        help="Optional JSON path describing the measured/replayed prepared inference state tensor footprint",
    )
    parser.add_argument(
        "--prepare-slim-no-raw-state-vision",
        action="store_true",
        help="Drop gen_data_clean.raw_state_vision from the prepared state; keeps latent/noise/mask state",
    )
    parser.add_argument(
        "--prepare-slim-derive-condition-reference",
        action="store_true",
        help="Do not store condition_reference in prepared state; derive it from x0_tokens on replay/cache hit",
    )
    parser.add_argument(
        "--prepare-slim-derive-initial-noise",
        action="store_true",
        help="Do not store initial_noise in prepared state; derive it from seed/x0_tokens/mask on replay/cache hit",
    )
    parser.add_argument(
        "--backend",
        choices=("official", "official_action_only"),
        default="official",
        help="official runs the upstream Cosmos output path; official_action_only skips inverse-dynamics vision decode/save",
    )
    args = parser.parse_args()

    model = flash_rt.load_model(
        args.checkpoint,
        framework="torch",
        config="cosmos3_edge",
        hardware="thor",
    )
    model.set_prompt(input_json=args.input_json)
    out = model.infer(
        output_dir=args.output_dir,
        seed=args.seed,
        backend=args.backend,
        vae_path=args.vae_path,
        cosmos_root=args.cosmos_root,
        config_file=args.config_file,
        live_dump_out=args.live_dump_out,
        live_flashrt_handoff=args.live_flashrt_handoff,
        live_boundary_out=args.live_boundary_out,
        live_boundary_in=args.live_boundary_in,
        live_boundary_prepare_in=args.live_boundary_prepare_in,
        live_boundary_prepare_live=args.live_boundary_prepare_live,
        live_handoff_trace_out=args.live_handoff_trace_out,
        upstream_trace_out=args.upstream_trace_out,
        vae_encode_dump_out=args.vae_encode_dump_out,
        vae_latent_in=args.vae_latent_in,
        vae_encode_dump_input=args.vae_encode_dump_input,
        vae_encode_profile_out=args.vae_encode_profile_out,
        vae_native_rms_silu=args.vae_native_rms_silu,
        vae_t1_conv2d=args.vae_t1_conv2d,
        vae_native_avgdown3d=args.vae_native_avgdown3d,
        vae_channels_last3d_conv320=args.vae_channels_last3d_conv320,
        vae_compile_encode=args.vae_compile_encode,
        vae_compile_trace_out=args.vae_compile_trace_out,
        prepare_dump_out=args.prepare_dump_out,
        prepare_replay_in=args.prepare_replay_in,
        prepare_inventory_out=args.prepare_inventory_out,
        prepare_slim_no_raw_state_vision=args.prepare_slim_no_raw_state_vision,
        prepare_slim_derive_condition_reference=args.prepare_slim_derive_condition_reference,
        prepare_slim_derive_initial_noise=args.prepare_slim_derive_initial_noise,
        live_prelayer_bootstrap=args.live_prelayer_bootstrap,
        live_warm_request=args.live_warm_request,
        cache_warmup_vae=args.cache_warmup_vae,
        cache_warmup_prepare=args.cache_warmup_prepare,
        warmup=args.warmup,
    )
    print(f"[cosmos3_edge] {args.backend} wall={out['wall_s']:.2f}s")
    print(f"[cosmos3_edge] output_dir={out['output_dir']}")
    for path in out.get("sample_outputs", []):
        print(f"[cosmos3_edge] sample_outputs={path}")
    if "live_dump_out" in out:
        print(f"[cosmos3_edge] live_dump_out={out['live_dump_out']}")
    if out.get("live_flashrt_handoff"):
        print("[cosmos3_edge] live_flashrt_handoff=true")
    if out.get("live_prelayer_bootstrap"):
        print("[cosmos3_edge] live_prelayer_bootstrap=true")
    if out.get("live_warm_request"):
        print("[cosmos3_edge] live_warm_request=true")
    if out.get("cache_warmup_vae"):
        print("[cosmos3_edge] cache_warmup_vae=true")
    if out.get("cache_warmup_prepare"):
        print("[cosmos3_edge] cache_warmup_prepare=true")
    if "warmup" in out:
        print(f"[cosmos3_edge] warmup={out['warmup']}")
    if "live_boundary_out" in out:
        print(f"[cosmos3_edge] live_boundary_out={out['live_boundary_out']}")
    if "live_boundary_in" in out:
        print(f"[cosmos3_edge] live_boundary_in={out['live_boundary_in']}")
    if "live_boundary_prepare_in" in out:
        print(f"[cosmos3_edge] live_boundary_prepare_in={out['live_boundary_prepare_in']}")
    if out.get("live_boundary_prepare_live"):
        print("[cosmos3_edge] live_boundary_prepare_live=true")
    if "live_handoff_trace_out" in out:
        print(f"[cosmos3_edge] live_handoff_trace_out={out['live_handoff_trace_out']}")
    if "upstream_trace_out" in out:
        print(f"[cosmos3_edge] upstream_trace_out={out['upstream_trace_out']}")
    if "vae_encode_dump_out" in out:
        print(f"[cosmos3_edge] vae_encode_dump_out={out['vae_encode_dump_out']}")
    if "vae_latent_in" in out:
        print(f"[cosmos3_edge] vae_latent_in={out['vae_latent_in']}")
    if out.get("vae_encode_dump_input"):
        print("[cosmos3_edge] vae_encode_dump_input=true")
    if "vae_encode_profile_out" in out:
        print(f"[cosmos3_edge] vae_encode_profile_out={out['vae_encode_profile_out']}")
    if out.get("vae_native_rms_silu"):
        print("[cosmos3_edge] vae_native_rms_silu=true")
    if out.get("vae_t1_conv2d"):
        print("[cosmos3_edge] vae_t1_conv2d=true")
    if out.get("vae_native_avgdown3d"):
        print("[cosmos3_edge] vae_native_avgdown3d=true")
    if out.get("vae_channels_last3d_conv320"):
        print("[cosmos3_edge] vae_channels_last3d_conv320=true")
    if out.get("vae_compile_encode"):
        print("[cosmos3_edge] vae_compile_encode=true")
    if "vae_compile_trace_out" in out:
        print(f"[cosmos3_edge] vae_compile_trace_out={out['vae_compile_trace_out']}")
    if "prepare_dump_out" in out:
        print(f"[cosmos3_edge] prepare_dump_out={out['prepare_dump_out']}")
    if "prepare_replay_in" in out:
        print(f"[cosmos3_edge] prepare_replay_in={out['prepare_replay_in']}")
    if "prepare_inventory_out" in out:
        print(f"[cosmos3_edge] prepare_inventory_out={out['prepare_inventory_out']}")
    if out.get("prepare_slim_no_raw_state_vision"):
        print("[cosmos3_edge] prepare_slim_no_raw_state_vision=true")
    if out.get("prepare_slim_derive_condition_reference"):
        print("[cosmos3_edge] prepare_slim_derive_condition_reference=true")
    if out.get("prepare_slim_derive_initial_noise"):
        print("[cosmos3_edge] prepare_slim_derive_initial_noise=true")


if __name__ == "__main__":
    main()
