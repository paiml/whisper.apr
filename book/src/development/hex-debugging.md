# Debugging Models with Hex Dumps

This chapter teaches you how to use hex dumps to debug ML model files. When transcription fails silently or produces garbage, the problem is often in the model file itself - corrupted weights, wrong format, or misaligned data.

## What is a Hex Dump?

A hex dump shows raw file bytes as hexadecimal numbers (0-9, A-F). Each byte is two hex digits.

```
Offset    Hex bytes (16 per line)                        ASCII
00000000  41 50 52 31 00 00 00 01 74 69 6e 79 00 00 00   APR1....tiny...
00000010  80 01 00 00 06 00 00 00 04 00 00 00 3f 80 00   ............?...
```

**Reading hex dumps:**
- **Offset** (left): Position in file (bytes from start)
- **Hex bytes** (middle): Raw data as hex pairs
- **ASCII** (right): Printable characters (dots for unprintable)

## Quick Reference: Common Values

| Hex | Decimal | Float | Meaning |
|-----|---------|-------|---------|
| `00 00 00 00` | 0 | 0.0 | Zero |
| `3f 80 00 00` | 1065353216 | 1.0 | One (IEEE 754) |
| `bf 80 00 00` | 3212836864 | -1.0 | Negative one |
| `7f 80 00 00` | 2139095040 | +Inf | Infinity (BAD!) |
| `ff 80 00 00` | 4286578688 | -Inf | Neg infinity (BAD!) |
| `7f c0 00 00` | 2143289344 | NaN | Not a number (BAD!) |

## Tools

### xxd (recommended)

```bash
# First 256 bytes
xxd model.apr | head -16

# Specific offset (skip 1024 bytes, show 64)
xxd -s 1024 -l 64 model.apr

# Just hex, no ASCII
xxd -p model.apr | head -1
```

### hexdump

```bash
# Canonical format
hexdump -C model.apr | head -20

# 4-byte integers (little-endian)
hexdump -e '"%08x: " 4/4 "%08x " "\n"' model.apr | head -10
```

### od (octal dump, but supports hex)

```bash
# Show as floats
od -A x -t f4 -N 64 model.apr
```

## Model File Anatomy

### APR Format Header

```
Offset  Size  Field           Example
------  ----  -----           -------
0x0000  4     Magic           "APR1" (41 50 52 31)
0x0004  4     Version         0x00000001
0x0008  4     Model type      0x00000000 = tiny
0x000C  4     n_vocab         51865 (0x0000CA99)
0x0010  4     n_audio_ctx     1500
0x0014  4     n_audio_state   384 (tiny), 512 (base)
0x0018  4     n_audio_head    6 (tiny), 8 (base)
0x001C  4     n_audio_layer   4 (tiny), 6 (base)
0x0020  4     n_text_ctx      448
0x0024  4     n_text_state    384 (tiny), 512 (base)
0x0028  4     n_text_head     6 (tiny), 8 (base)
0x002C  4     n_text_layer    4 (tiny), 6 (base)
0x0030  4     n_mels          80
0x0034  4     ftype           0=f32, 1=f16, 2=q8_0
```

### Verifying the Header

```bash
# Check magic bytes
xxd -l 4 model.apr
# Should show: 00000000: 4150 5231  APR1

# Check model dimensions (tiny = 384 = 0x180)
xxd -s 0x14 -l 4 model.apr
# Should show: 00000014: 8001 0000  (little-endian 0x00000180 = 384)
```

## Common Problems and How to Find Them

### Problem 1: Wrong Magic Bytes

**Symptom**: "Invalid model format" error

```bash
xxd -l 4 model.apr
```

**Good**: `4150 5231` (APR1)
**Bad**: `4747 4d4c` (GGML - wrong format)
**Bad**: `0000 0000` (corrupted/empty)

### Problem 2: NaN/Inf in Weights

**Symptom**: Transcription outputs garbage or empty text

```bash
# Search for NaN pattern (7f c0 00 00 or 7f c? ?? ??)
xxd model.apr | grep -E "7fc[0-9a-f]"

# Search for Inf pattern
xxd model.apr | grep "7f80 0000"
```

**If found**: Model is corrupted. Re-download or re-convert.

### Problem 3: All Zeros

**Symptom**: Model loads but produces no output

```bash
# Check weight section (after header, ~0x100)
xxd -s 256 -l 128 model.apr

# Count zero bytes
xxd -p model.apr | grep -o "00" | wc -l
```

**If mostly zeros**: Incomplete download or failed conversion.

### Problem 4: Wrong Endianness

**Symptom**: Dimensions look wrong (e.g., 384 appears as 2550136832)

Little-endian (correct for x86/WASM):
```
384 = 0x00000180 → stored as: 80 01 00 00
```

Big-endian (wrong):
```
384 = 0x00000180 → stored as: 00 00 01 80
```

```bash
# Check dimension at offset 0x14
xxd -s 0x14 -l 4 model.apr
# Correct (tiny): 8001 0000
# Wrong endian:   0000 0180
```

### Problem 5: Truncated File

**Symptom**: Crashes during load or missing layers

```bash
# Check file size
ls -l model.apr

# Expected sizes (approximate):
# tiny:   ~75 MB
# base:   ~145 MB
# small:  ~465 MB
```

```bash
# Check last bytes (should be valid floats, not zeros)
xxd model.apr | tail -5
```

## Debugging the Weight Tensor Layout

### Finding a Specific Tensor

Tensors are stored sequentially. Use the tensor index to find offsets:

```bash
# List tensors with our tool
cargo run --example list_model_tensors -- model.apr
```

Output:
```
Tensor 0: encoder.conv1.weight [384, 80, 3] @ offset 0x100
Tensor 1: encoder.conv1.bias [384] @ offset 0x1C500
Tensor 2: encoder.conv2.weight [384, 384, 3] @ offset 0x1C700
...
```

### Inspecting a Tensor

```bash
# Dump first 64 bytes of conv1 weights
xxd -s 0x100 -l 64 model.apr
```

### Verifying Weight Statistics

Good weights have varied values:
```
3e99 999a 3f00 0000 be4c cccd 3f4c cccd ...
(0.3)     (0.5)     (-0.2)    (0.8)
```

Bad weights (stuck/dead):
```
0000 0000 0000 0000 0000 0000 0000 0000 ...
(all zeros - layer is dead)
```

## Comparing Two Models

### Byte-for-Byte Diff

```bash
# Full comparison (slow for large files)
cmp model_a.apr model_b.apr

# Show first difference
cmp -l model_a.apr model_b.apr | head -1
```

### Header Comparison

```bash
# Compare headers
xxd -l 64 model_a.apr > /tmp/a.hex
xxd -l 64 model_b.apr > /tmp/b.hex
diff /tmp/a.hex /tmp/b.hex
```

### Statistical Comparison

```bash
# Compare checksums of weight sections
tail -c +257 model_a.apr | md5sum
tail -c +257 model_b.apr | md5sum
```

## Practical Debugging Session

**Scenario**: Transcription produces only whitespace.

**Step 1**: Verify header
```bash
xxd -l 64 model.apr
```
✓ Magic is "APR1", dimensions look correct.

**Step 2**: Check for NaN/Inf
```bash
xxd model.apr | grep -E "(7fc|7f80|ff80)" | head -5
```
✗ Found `7fc00000` at offset 0x1234 - NaN detected!

**Step 3**: Identify corrupted tensor
```bash
cargo run --example list_model_tensors -- model.apr | grep -B1 "0x1234"
```
→ Corrupted tensor: `encoder.blocks.0.attn.out_proj.weight`

**Step 4**: Root cause
```bash
pmat five-whys "NaN in encoder.blocks.0.attn.out_proj.weight"
```

**Step 5**: Fix
- Re-download model from source
- Or re-run conversion with `--check-nan` flag

## Hex Dump Cheat Sheet

| Task | Command |
|------|---------|
| View header | `xxd -l 64 model.apr` |
| View offset | `xxd -s 0x1000 -l 32 model.apr` |
| Find NaN | `xxd model.apr \| grep "7fc0"` |
| Find Inf | `xxd model.apr \| grep "7f80 0000"` |
| Compare files | `cmp model_a.apr model_b.apr` |
| File size | `ls -lh model.apr` |
| Checksum | `md5sum model.apr` |
| View as floats | `od -A x -t f4 -N 64 model.apr` |
| Binary to hex | `xxd -p model.apr \| head -1` |
| Hex to binary | `xxd -r -p input.hex output.bin` |

## Float Encoding Reference

IEEE 754 single-precision (32-bit) format:

```
Bit:  31  30-23   22-0
      S   EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM
      │   │        └─ Mantissa (23 bits)
      │   └─ Exponent (8 bits, biased by 127)
      └─ Sign (0=positive, 1=negative)
```

**Examples**:
```
1.0  = 0 01111111 00000000000000000000000 = 3F800000
-1.0 = 1 01111111 00000000000000000000000 = BF800000
0.5  = 0 01111110 00000000000000000000000 = 3F000000
2.0  = 0 10000000 00000000000000000000000 = 40000000
```

**Special values**:
```
+0   = 00000000
-0   = 80000000
+Inf = 7F800000
-Inf = FF800000
NaN  = 7FC00000 (quiet NaN)
```

## Summary

1. **Always check the header first** - wrong magic means wrong format
2. **Search for NaN/Inf** - these corrupt the entire forward pass
3. **Verify file size** - truncated files cause mysterious crashes
4. **Compare against known-good** - when in doubt, diff it
5. **Use the right endianness** - x86/WASM is little-endian

When all else fails:
```bash
xxd model.apr | less
```
And start reading bytes.
