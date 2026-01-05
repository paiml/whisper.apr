# Whisper.apr Nix Flake
# WAPR-PERF-002: Reproducibility Infrastructure
#
# Usage:
#   nix develop    # Enter development shell
#   nix build      # Build the project
#   nix flake check # Run tests
{
  description = "WASM-first automatic speech recognition engine implementing OpenAI Whisper";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          targets = [ "wasm32-unknown-unknown" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain
            rustToolchain

            # Build tools
            pkg-config
            openssl

            # WASM tools
            wasm-pack
            wasm-bindgen-cli

            # Development tools
            cargo-watch
            cargo-edit
            cargo-nextest
            cargo-llvm-cov

            # Optional: for benchmarking
            gnuplot

            # Optional: for documentation
            mdbook
          ];

          shellHook = ''
            echo "whisper.apr development environment"
            echo "Rust: $(rustc --version)"
            echo ""
            echo "Commands:"
            echo "  cargo build              - Build native"
            echo "  cargo build --target wasm32-unknown-unknown - Build WASM"
            echo "  cargo test               - Run tests"
            echo "  cargo bench              - Run benchmarks"
            echo "  make coverage            - Generate coverage report"
          '';

          # Environment variables for reproducibility
          RUST_BACKTRACE = "1";
          CARGO_NET_GIT_FETCH_WITH_CLI = "true";
        };

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "whisper-apr";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = with pkgs; [ pkg-config ];
          buildInputs = with pkgs; [ openssl ];
        };
      }
    );
}
