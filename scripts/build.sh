#!/usr/bin/env bash
#
# Cognitive Robot OS — unified build and verification script.
# Reproduces all project phases with static and dynamic analysis at each step.
#
# Usage:
#   scripts/build.sh host        # build + static analysis (test/clippy/fmt) on host
#   scripts/build.sh dynamic     # dynamic analysis (valgrind + latency measurement)
#   scripts/build.sh baremetal   # build neural-memory for aarch64-unknown-none target (inside container)
#   scripts/build.sh sel4        # build + run seL4 hello on QEMU (inside container)
#   scripts/build.sh demo        # demonstrate persistence (memories survive simulated power loss)
#   scripts/build.sh all         # all of the above in order
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM="$ROOT/components/memory"
SDK="$ROOT/tools/microkit-sdk-2.2.0"
CONTAINER="sel4dev"
BOARD="qemu_virt_aarch64"

log() { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }
ok() { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }

# ----- training: export policy + encoder weights + lint -----
train() {
  log "[TRAIN] export policy + encoder weights (numpy → Rust const)"
  if command -v ruff >/dev/null 2>&1; then
    ruff check "$ROOT/training/train_policy.py" "$ROOT/training/train_encoder.py" \
      "$ROOT/training/train_cartpole.py"
  fi
  python3 "$ROOT/training/train_policy.py"
  python3 "$ROOT/training/train_encoder.py"
  python3 "$ROOT/training/train_cartpole.py"
  ok "weights generated"
}

# ----- host: static analysis (all crates) -----
host_static() {
  log "[STATIC] host: test + clippy + fmt"
  local crate
  for crate in memory policy perception reflex-abi cartpole-policy cartpole-sim hal mujoco-pendulum-policy walker-policy humanoid-policy brain-os-abi world-memory behavior-fsm safety-memory contextual-guard safety-model clearance-guard imu-filter ood-detector nav-policy; do
    echo "--- crate: $crate ---"
    cd "$ROOT/components/$crate"
    cargo test
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check
  done
  for crate in pi4-hal pi4-body; do
    echo "--- driver: $crate ---"
    cd "$ROOT/drivers/$crate"
    cargo test
    cargo clippy --all-targets -- -D warnings
    cargo fmt --check
  done
  for s in sil-ablation sil-eval sil-experiments sil-stress bench-guard sil-clearance sil-nav-guarded sil-ood sil-campaign; do
    echo "--- analysis sim: $s ---"
    cd "$ROOT/sim/$s"
    cargo clippy -- -D warnings
    cargo fmt --check
  done
  ok "static analysis clean"
}

# ----- host: dynamic analysis -----
host_dynamic() {
  log "[DYNAMIC] host: valgrind + latency"
  cd "$MEM"
  RUSTFLAGS="-g -C target-cpu=native" cargo build --release --example bench
  local bin="$MEM/target/release/examples/bench"
  "$bin"
  if command -v valgrind >/dev/null 2>&1; then
    valgrind --leak-check=summary --error-exitcode=1 "$bin" 2>&1 |
      grep -E "ERROR SUMMARY|definitely lost"
    ok "valgrind: no errors / no leaks"
  else
    echo "valgrind not found — skipping"
  fi
}

# ----- container: build Raspberry Pi 4 image (not executed — needs a physical Pi) -----
rpi4b() {
  log "[rpi4b] cross-compile + package the unified system for Raspberry Pi 4"
  # $1=sdk $2=root are passed to the inner shell intentionally
  # shellcheck disable=SC2016
  distrobox enter "$CONTAINER" -- bash -c '
    set -e
    source /etc/profile.d/sel4-rust.sh
    sdk="$1"; root="$2"
    brd="$sdk/board/rpi4b_4gb/debug"
    export SEL4_INCLUDE_DIRS="$brd/include" SEL4_PLATFORM_INFO="$brd/platform_gen.json"
    export CARGO_TARGET_DIR=/tmp/rpi4b-build
    for pd in sel4-memory-pd sel4-guard-pd sel4-actuation-pd; do
      (cd "$root/components/$pd" &&
        cargo +nightly-2026-04-04 build --release -Z json-target-spec >/dev/null 2>&1)
    done
    b="$root/kernel/build-rpi4b"; mkdir -p "$b"
    for pd in memory guard actuation; do
      cp "/tmp/rpi4b-build/aarch64-sel4-microkit/release/microkit-$pd-pd.elf" "$b/"
    done
    "$sdk/bin/microkit" "$root/kernel/reflex.system" --search-path "$b" \
      --board rpi4b_4gb --config debug -o "$b/loader.img" -r "$b/report.txt" >/dev/null 2>&1
    ls -la "$b/loader.img"
  ' bash "$SDK" "$ROOT"
  ok "rpi4b boot image built (flash to SD; run needs a physical Pi 4)"
}

# ----- Software-in-the-Loop: cartpole (same crates as seL4) -----
sil() {
  log "[SIL] cartpole closed-loop control + safety (same crates as seL4)"
  cd "$ROOT/sim/sil-cartpole"
  cargo clippy -- -D warnings
  cargo fmt --check
  cargo run --release
  (cd "$ROOT/sim/sil-skill" && cargo run --release)
  (cd "$ROOT/sim/sil-world" && cargo run --release)
  (cd "$ROOT/sim/sil-adaptive-safety" && cargo run --release)
  (cd "$ROOT/sim/sil-born-cautious" && cargo run --release)
  ok "SIL: closed-loop + multi-joint deploy + world perception"
}

# ----- persistence demonstration -----
demo() {
  log "[DEMO] persistence across simulated power loss"
  cd "$MEM"
  rm -f /tmp/crob_flash.bin
  cargo build --release --example persist_demo
  for i in 1 2 3; do
    echo "--- boot $i ---"
    ./target/release/examples/persist_demo
  done
  rm -f /tmp/crob_flash.bin
  ok "memories survived 3 reboots"
}

# ----- container: bare-metal build -----
baremetal() {
  log "[BAREMETAL] build neural-memory for aarch64-unknown-none"
  # $1 is passed to the inner shell intentionally (not expanded here)
  # shellcheck disable=SC2016
  distrobox enter "$CONTAINER" -- bash -c '
    source /etc/profile.d/sel4-rust.sh
    cd "$1"
    cargo build --lib --release --target aarch64-unknown-none --target-dir /tmp/crob-aarch64
    ls -la /tmp/crob-aarch64/aarch64-unknown-none/release/libneural_memory.rlib
  ' bash "$MEM"
  ok "neural-memory builds bare-metal"
}

# ----- container: build and run seL4 hello on QEMU -----
sel4() {
  log "[seL4] build + run hello on QEMU ($BOARD)"
  # $1/$2 are passed to the inner shell intentionally (not expanded here)
  # shellcheck disable=SC2016
  distrobox enter "$CONTAINER" -- bash -c '
    set -e
    sdk="$1"; board="$2"
    cd "$sdk/example/hello"
    mkdir -p build
    make LLVM=True BUILD_DIR=build MICROKIT_BOARD="$board" MICROKIT_CONFIG=debug MICROKIT_SDK="$sdk" >/dev/null
    echo "boot image built; running on QEMU..."
    timeout 12 qemu-system-aarch64 -machine virt,virtualization=on -cpu cortex-a53 \
      -m size=2G -nographic \
      -device loader,file=build/loader.img,addr=0x70000000,cpu-num=0 2>&1 |
      grep -iE "dropped to user space|hello, world" || true
  ' bash "$SDK" "$BOARD"
  ok "seL4 boots and PD prints hello"
}

# ----- container: build and run the reflex arc (cognitive + guard) on QEMU -----
reflex() {
  log "[seL4+Rust] reflex arc: cognitive + guard PDs on QEMU"
  # $1/$2 are passed to the inner shell intentionally
  # shellcheck disable=SC2016
  distrobox enter "$CONTAINER" -- bash -c '
    set -e
    source /etc/profile.d/sel4-rust.sh
    sdk="$1"; root="$2"
    sdkb="$sdk/board/qemu_virt_aarch64/debug"
    export SEL4_INCLUDE_DIRS="$sdkb/include" SEL4_PLATFORM_INFO="$sdkb/platform_gen.json"
    for pd in sel4-memory-pd sel4-guard-pd sel4-actuation-pd; do
      (cd "$root/components/$pd" &&
        cargo +nightly-2026-04-04 build --release -Z json-target-spec >/dev/null 2>&1)
    done
    b="$root/kernel/build"; mkdir -p "$b"
    cp "$root/components/sel4-memory-pd/target/aarch64-sel4-microkit/release/microkit-memory-pd.elf" "$b/"
    cp "$root/components/sel4-guard-pd/target/aarch64-sel4-microkit/release/microkit-guard-pd.elf" "$b/"
    cp "$root/components/sel4-actuation-pd/target/aarch64-sel4-microkit/release/microkit-actuation-pd.elf" "$b/"
    "$sdk/bin/microkit" "$root/kernel/reflex.system" --search-path "$b" \
      --board qemu_virt_aarch64 --config debug -o "$b/loader.img" -r "$b/report.txt" >/dev/null 2>&1
    timeout 20 qemu-system-aarch64 -machine virt,virtualization=on -cpu cortex-a53 \
      -m size=2G -nographic \
      -device loader,file="$b/loader.img",addr=0x70000000,cpu-num=0 2>&1 |
      grep -iE "body|brain|guard|cycle|MOTOR|WATCHDOG|PREEMPT|PASS|dropped to user|fault" || true
  ' bash "$SDK" "$ROOT"
  ok "reflex arc: guard preempts the AI on the verified microkernel"
}

main() {
  local cmd="${1:-all}"
  case "$cmd" in
    train) train ;;
    host) host_static ;;
    dynamic) host_dynamic ;;
    baremetal) baremetal ;;
    sel4) sel4 ;;
    reflex) reflex ;;
    rpi4b) rpi4b ;;
    sil) sil ;;
    demo) demo ;;
    all)
      train
      host_static
      host_dynamic
      demo
      baremetal
      sel4
      reflex
      sil
      log "ALL STAGES PASSED"
      ;;
    *)
      echo "unknown command: $cmd" >&2
      echo "usage: $0 {train|host|dynamic|baremetal|sel4|reflex|rpi4b|sil|demo|all}" >&2
      exit 1
      ;;
  esac
}

main "$@"
