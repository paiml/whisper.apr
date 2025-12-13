# Whisper.apr Book

Documentation for Whisper.apr, a WASM-first speech recognition engine.

## Building the Book

Requires [mdBook](https://rust-lang.github.io/mdBook/):

```bash
# Install mdBook
cargo install mdbook

# Build the book
cd book
mdbook build

# Serve locally with live reload
mdbook serve --open
```

## Structure

```
book/
├── book.toml          # mdBook configuration
├── src/
│   ├── SUMMARY.md     # Table of contents
│   ├── introduction.md
│   ├── getting-started/
│   ├── architecture/
│   ├── api-reference/
│   ├── performance/
│   ├── examples/
│   ├── development/
│   ├── advanced/
│   └── appendix/
└── book/              # Generated output (git-ignored)
```

## Contributing

1. Edit Markdown files in `src/`
2. Run `mdbook serve` to preview
3. Submit PR with changes

## Deployment

The book is automatically deployed to GitHub Pages on merge to `main`.
