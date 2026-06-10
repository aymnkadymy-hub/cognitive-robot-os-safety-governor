# Environment & Setup

This documents the exact environment in which the committed reference results were produced, and
how to recreate it. The **software-in-the-loop (SiL) evaluation needs only a Rust toolchain** — no
GPU, no network, no physical hardware. The seL4/QEMU path is optional and only needed to reproduce
the on-microkernel reflex-arc demonstration.

## 1. Host toolchain (required for all results)

The reference outputs were generated with:

```
rustc 1.96.0 (ac68faa20 2026-05-25)
cargo 1.96.0 (30a34c682 2026-05-25)
```

Install Rust via [rustup](https://rustup.rs) (any reasonably recent stable toolchain works; the
results are deterministic and toolchain-independent except for absolute timing numbers):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version && cargo --version
```

Optional host analysis tools used by `scripts/build.sh` (not needed just to reproduce results):
`clippy`, `rustfmt` (rustup components), `valgrind`, `heaptrack`, `perf`, `shellcheck`.

## 2. Determinism (why everyone gets the same numbers)

- All randomness comes from a **fixed-seed linear congruential generator** (`Rng(seed)`) inside
  each experiment — never from `rand`, the wall clock, or the OS.
- All safety crates are `no_std`, heap-free, with no global mutable state → bit-for-bit repeatable.
- The evaluation uses seeds `1..=40`; **every (method × environment) cell sees the same worlds**
  (a *paired* comparison).
- **Only exception:** absolute timing numbers (`bench-guard`) depend on the host (clock/cache/OS).
  The **ratios** and ordering are stable, and the code is `O(1)` and allocation-free.

## 3. Optional: seL4 / Microkit (QEMU) environment

Only required for `scripts/build.sh sel4` / `reflex` (boot the Guard + Cognitive PDs on the
verified microkernel under QEMU). The seL4 toolchain is isolated from the host using a container.

### 3a. The Microkit SDK is **not bundled** (it is ~600 MB and freely downloadable)
```sh
mkdir -p tools && cd tools
curl -sL -o microkit-sdk.tar.gz \
  https://github.com/seL4/microkit/releases/download/2.2.0/microkit-sdk-2.2.0-linux-x86-64.tar.gz
tar xzf microkit-sdk.tar.gz       # → tools/microkit-sdk-2.2.0/
```
Supports `qemu_virt_aarch64` (simulation) and `rpi4b_1gb/2gb/4gb/8gb` (future hardware).

### 3b. Isolated build container (Ubuntu 22.04 via distrobox/podman)
```sh
sudo pacman -S distrobox podman          # (or your distro's equivalent)
distrobox create --image docker.io/library/ubuntu:22.04 --name sel4dev --yes
distrobox enter sel4dev -- bash -c '
  sudo apt-get update
  sudo apt-get install -y build-essential curl clang lld qemu-system-arm pkg-config
  sudo mkdir -p /usr/local/{rustup,cargo} && sudo chown -R $USER /usr/local/{rustup,cargo}
  export RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
  curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
     --no-modify-path --default-toolchain nightly --profile minimal
  export PATH=/usr/local/cargo/bin:$PATH
  rustup target add aarch64-unknown-none
  rustup component add rust-src
  echo "export RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:\$PATH" \
     | sudo tee /etc/profile.d/sel4-rust.sh
'
```

### 3c. Run seL4 "hello" on QEMU (sanity check)
```sh
distrobox enter sel4dev
cd tools/microkit-sdk-2.2.0/example/hello && mkdir -p build
make LLVM=True BUILD_DIR=build MICROKIT_BOARD=qemu_virt_aarch64 \
     MICROKIT_CONFIG=debug MICROKIT_SDK=$(pwd)/../..
qemu-system-aarch64 -machine virt,virtualization=on -cpu cortex-a53 -m size=2G \
  -nographic -device loader,file=build/loader.img,addr=0x70000000,cpu-num=0
# exit QEMU with Ctrl-A then X
```

## 4. Optional: Python (training only)

The RL policies under [`training/`](../training/) are trained host-side (PyTorch/Stable-Baselines
style) and exported to ONNX, then baked into Rust `weights.rs` constants. **You do not need Python
to reproduce any safety result** — the trained weights are already committed in the crates.
Training is provided for completeness/transparency.

## 5. Formal-verification tooling (optional)

Property tests run under plain `cargo test`. The `#[cfg(kani)]` proof harnesses require the
[Kani Rust verifier](https://model-checking.github.io/kani/); they are optional — the same
invariants are also covered by `proptest`-based property tests that run with stock `cargo test`.
See [`VERIFICATION.md`](VERIFICATION.md).
