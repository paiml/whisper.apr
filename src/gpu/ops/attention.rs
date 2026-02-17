//! GPU-accelerated attention computation (WAPR-GPU-ATTENTION-001)
//!
//! Implements scaled dot-product attention on GPU:
//! - Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//!
//! Optimizations:
//! - Tiled matrix multiplication for Q @ K^T
//! - Fused scaling with 1/sqrt(d_k)
//! - Online softmax for numerical stability
//! - Optional causal masking for decoder attention

use super::matmul::TileSize;

/// Attention configuration
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: u32,
    /// Head dimension (d_k = d_model / num_heads)
    pub head_dim: u32,
    /// Sequence length for queries
    pub seq_len_q: u32,
    /// Sequence length for keys/values
    pub seq_len_kv: u32,
    /// Whether to apply causal mask (for decoder self-attention)
    pub causal: bool,
    /// Tile size for matrix multiplication
    pub tile_size: TileSize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64, // 512 / 8 for base model
            seq_len_q: 1500,
            seq_len_kv: 1500,
            causal: false,
            tile_size: TileSize::Tile16x16,
        }
    }
}

impl AttentionConfig {
    /// Create config for Whisper tiny model
    #[must_use]
    pub fn whisper_tiny() -> Self {
        Self {
            num_heads: 6,
            head_dim: 64, // 384 / 6
            ..Default::default()
        }
    }

    /// Create config for Whisper base model
    #[must_use]
    pub fn whisper_base() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64, // 512 / 8
            ..Default::default()
        }
    }

    /// Create config for Whisper small model
    #[must_use]
    pub fn whisper_small() -> Self {
        Self {
            num_heads: 12,
            head_dim: 64, // 768 / 12
            ..Default::default()
        }
    }

    /// Create config for decoder self-attention (causal)
    #[must_use]
    pub fn decoder_self_attention(num_heads: u32, head_dim: u32, seq_len: u32) -> Self {
        Self {
            num_heads,
            head_dim,
            seq_len_q: seq_len,
            seq_len_kv: seq_len,
            causal: true,
            tile_size: TileSize::Tile16x16,
        }
    }

    /// Create config for cross-attention (encoder-decoder)
    #[must_use]
    pub fn cross_attention(
        num_heads: u32,
        head_dim: u32,
        decoder_len: u32,
        encoder_len: u32,
    ) -> Self {
        Self {
            num_heads,
            head_dim,
            seq_len_q: decoder_len,
            seq_len_kv: encoder_len,
            causal: false,
            tile_size: TileSize::Tile16x16,
        }
    }

    /// Get the scaling factor (1 / sqrt(d_k))
    #[must_use]
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Total dimension (d_model = num_heads * head_dim)
    #[must_use]
    pub fn d_model(&self) -> u32 {
        self.num_heads * self.head_dim
    }
}

/// GPU attention operation
#[derive(Debug, Clone)]
pub struct GpuAttention {
    config: AttentionConfig,
}

impl GpuAttention {
    /// Create new GPU attention operation
    #[must_use]
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Generate WGSL shader for attention scores (Q @ K^T / sqrt(d_k))
    ///
    /// This computes the attention weights before softmax.
    #[must_use]
    pub fn generate_scores_shader(&self) -> String {
        let tile = self.config.tile_size.dimension();
        let scale = self.config.scale();
        let seq_q = self.config.seq_len_q;
        let seq_kv = self.config.seq_len_kv;
        let head_dim = self.config.head_dim;

        format!(
            r#"// Attention Scores: Q @ K^T / sqrt(d_k)
// Generated for WAPR-GPU-ATTENTION-001

struct Params {{
    seq_len_q: u32,
    seq_len_kv: u32,
    head_dim: u32,
    scale: f32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;      // [seq_q, head_dim]
@group(0) @binding(2) var<storage, read> k: array<f32>;      // [seq_kv, head_dim]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>; // [seq_q, seq_kv]

const TILE_SIZE: u32 = {tile}u;
const SEQ_Q: u32 = {seq_q}u;
const SEQ_KV: u32 = {seq_kv}u;
const HEAD_DIM: u32 = {head_dim}u;
const SCALE: f32 = {scale};

var<workgroup> tile_q: array<f32, {tile_sq}>;
var<workgroup> tile_k: array<f32, {tile_sq}>;

@compute @workgroup_size({tile}, {tile})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let row = global_id.x;  // Query position
    let col = global_id.y;  // Key position
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum: f32 = 0.0;

    // Tile over head_dim
    let num_tiles = (HEAD_DIM + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let k_idx = t * TILE_SIZE + local_col;

        // Load Q tile
        if (row < SEQ_Q && k_idx < HEAD_DIM) {{
            tile_q[local_row * TILE_SIZE + local_col] = q[row * HEAD_DIM + k_idx];
        }} else {{
            tile_q[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        // Load K^T tile (transposed access)
        if (col < SEQ_KV && k_idx < HEAD_DIM) {{
            tile_k[local_row * TILE_SIZE + local_col] = k[col * HEAD_DIM + k_idx];
        }} else {{
            tile_k[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        workgroupBarrier();

        // Compute partial dot product
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {{
            sum = sum + tile_q[local_row * TILE_SIZE + i] * tile_k[local_col * TILE_SIZE + i];
        }}

        workgroupBarrier();
    }}

    // Write scaled result
    if (row < SEQ_Q && col < SEQ_KV) {{
        scores[row * SEQ_KV + col] = sum * SCALE;
    }}
}}"#,
            tile = tile,
            seq_q = seq_q,
            seq_kv = seq_kv,
            head_dim = head_dim,
            scale = scale,
            tile_sq = tile * tile,
        )
    }

    /// Generate WGSL shader for causal masking
    ///
    /// Applies -inf to positions where query_pos < key_pos
    #[must_use]
    pub fn generate_causal_mask_shader(&self) -> String {
        let seq_q = self.config.seq_len_q;
        let seq_kv = self.config.seq_len_kv;

        format!(
            r#"// Causal Mask Application
// Sets scores[i,j] = -inf where i < j

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;

const SEQ_Q: u32 = {seq_q}u;
const SEQ_KV: u32 = {seq_kv}u;
const NEG_INF: f32 = -3.402823e+38;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    let total = SEQ_Q * SEQ_KV;

    if (idx >= total) {{
        return;
    }}

    let row = idx / SEQ_KV;
    let col = idx % SEQ_KV;

    // Causal mask: can only attend to positions <= current
    if (col > row) {{
        scores[idx] = NEG_INF;
    }}
}}"#,
            seq_q = seq_q,
            seq_kv = seq_kv,
        )
    }

    /// Generate WGSL shader for attention output (attn_weights @ V)
    #[must_use]
    pub fn generate_output_shader(&self) -> String {
        let tile = self.config.tile_size.dimension();
        let seq_q = self.config.seq_len_q;
        let seq_kv = self.config.seq_len_kv;
        let head_dim = self.config.head_dim;

        format!(
            r#"// Attention Output: softmax(scores) @ V
// Generated for WAPR-GPU-ATTENTION-001

@group(0) @binding(0) var<storage, read> weights: array<f32>;  // [seq_q, seq_kv] after softmax
@group(0) @binding(1) var<storage, read> v: array<f32>;        // [seq_kv, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [seq_q, head_dim]

const TILE_SIZE: u32 = {tile}u;
const SEQ_Q: u32 = {seq_q}u;
const SEQ_KV: u32 = {seq_kv}u;
const HEAD_DIM: u32 = {head_dim}u;

var<workgroup> tile_w: array<f32, {tile_sq}>;
var<workgroup> tile_v: array<f32, {tile_sq}>;

@compute @workgroup_size({tile}, {tile})
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let row = global_id.x;  // Query position
    let col = global_id.y;  // Head dimension
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum: f32 = 0.0;

    // Tile over seq_kv
    let num_tiles = (SEQ_KV + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {{
        let k_idx = t * TILE_SIZE + local_col;

        // Load weights tile
        if (row < SEQ_Q && k_idx < SEQ_KV) {{
            tile_w[local_row * TILE_SIZE + local_col] = weights[row * SEQ_KV + k_idx];
        }} else {{
            tile_w[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        // Load V tile
        let v_row = t * TILE_SIZE + local_row;
        if (v_row < SEQ_KV && col < HEAD_DIM) {{
            tile_v[local_row * TILE_SIZE + local_col] = v[v_row * HEAD_DIM + col];
        }} else {{
            tile_v[local_row * TILE_SIZE + local_col] = 0.0;
        }}

        workgroupBarrier();

        // Compute partial weighted sum
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {{
            sum = sum + tile_w[local_row * TILE_SIZE + i] * tile_v[i * TILE_SIZE + local_col];
        }}

        workgroupBarrier();
    }}

    // Write result
    if (row < SEQ_Q && col < HEAD_DIM) {{
        output[row * HEAD_DIM + col] = sum;
    }}
}}"#,
            tile = tile,
            seq_q = seq_q,
            seq_kv = seq_kv,
            head_dim = head_dim,
            tile_sq = tile * tile,
        )
    }

    /// Build the WGSL causal mask snippet (empty if non-causal attention)
    fn build_causal_check(&self) -> &'static str {
        if self.config.causal {
            r#"
        // Causal mask
        if (col > row) {
            score = NEG_INF;
        }"#
        } else {
            ""
        }
    }

    /// Build the WGSL header with bindings and constants for fused attention.
    fn build_fused_header(&self) -> String {
        let tile = self.config.tile_size.dimension();
        format!(
            r#"// Fused Multi-Head Attention
// Computes: softmax(Q @ K^T / sqrt(d_k) + mask) @ V

struct Params {{
    seq_len_q: u32,
    seq_len_kv: u32,
    head_dim: u32,
    num_heads: u32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const TILE_SIZE: u32 = {tile}u;
const SEQ_Q: u32 = {seq_q}u;
const SEQ_KV: u32 = {seq_kv}u;
const HEAD_DIM: u32 = {head_dim}u;
const SCALE: f32 = {scale};
const NEG_INF: f32 = -3.402823e+38;

var<workgroup> row_max: array<f32, {tile}>;
var<workgroup> row_sum: array<f32, {tile}>;"#,
            tile = tile,
            seq_q = self.config.seq_len_q,
            seq_kv = self.config.seq_len_kv,
            head_dim = self.config.head_dim,
            scale = self.config.scale(),
        )
    }

    /// Build the WGSL body: score computation, online softmax, and output accumulation.
    fn build_fused_body(&self) -> String {
        let tile = self.config.tile_size.dimension();
        let causal_check = self.build_causal_check();
        format!(
            r#"@compute @workgroup_size({tile}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let row = global_id.x;
    let head = group_id.y;
    let local_row = local_id.x;

    if (row >= SEQ_Q) {{ return; }}

    // Step 1: Compute attention scores and find max
    var max_score: f32 = NEG_INF;
    for (var col: u32 = 0u; col < SEQ_KV; col = col + 1u) {{
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < HEAD_DIM; d = d + 1u) {{
            score = score + q[(head * SEQ_Q + row) * HEAD_DIM + d] * k[(head * SEQ_KV + col) * HEAD_DIM + d];
        }}
        score = score * SCALE;
        {causal_check}
        max_score = max(max_score, score);
    }}
    row_max[local_row] = max_score;
    workgroupBarrier();

    // Step 2: Softmax weights and weighted V accumulation
    var sum_exp: f32 = 0.0;
    var out: array<f32, {head_dim}>;
    for (var d: u32 = 0u; d < HEAD_DIM; d = d + 1u) {{ out[d] = 0.0; }}

    for (var col: u32 = 0u; col < SEQ_KV; col = col + 1u) {{
        var score: f32 = 0.0;
        for (var d: u32 = 0u; d < HEAD_DIM; d = d + 1u) {{
            score = score + q[(head * SEQ_Q + row) * HEAD_DIM + d] * k[(head * SEQ_KV + col) * HEAD_DIM + d];
        }}
        score = score * SCALE;
        {causal_check}
        let weight = exp(score - max_score);
        sum_exp = sum_exp + weight;
        for (var d: u32 = 0u; d < HEAD_DIM; d = d + 1u) {{
            out[d] = out[d] + weight * v[(head * SEQ_KV + col) * HEAD_DIM + d];
        }}
    }}
    row_sum[local_row] = sum_exp;
    workgroupBarrier();

    // Step 3: Normalize and write output
    let norm = 1.0 / sum_exp;
    for (var d: u32 = 0u; d < HEAD_DIM; d = d + 1u) {{
        output[(head * SEQ_Q + row) * HEAD_DIM + d] = out[d] * norm;
    }}
}}"#,
            tile = tile,
            head_dim = self.config.head_dim,
            causal_check = causal_check,
        )
    }

    /// Generate complete fused attention shader.
    ///
    /// Combines Q@K^T scoring, optional causal mask, online softmax, and V accumulation
    /// into a single GPU dispatch for reduced memory bandwidth.
    #[must_use]
    pub fn generate_fused_attention_shader(&self) -> String {
        let header = self.build_fused_header();
        let body = self.build_fused_body();
        format!("{header}\n\n{body}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.d_model(), 512);
    }

    #[test]
    fn test_attention_config_whisper_tiny() {
        let config = AttentionConfig::whisper_tiny();
        assert_eq!(config.num_heads, 6);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.d_model(), 384);
    }

    #[test]
    fn test_attention_config_whisper_base() {
        let config = AttentionConfig::whisper_base();
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.d_model(), 512);
    }

    #[test]
    fn test_attention_config_whisper_small() {
        let config = AttentionConfig::whisper_small();
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.d_model(), 768);
    }

    #[test]
    fn test_attention_config_scale() {
        let config = AttentionConfig::default();
        let expected = 1.0 / 8.0; // 1 / sqrt(64)
        assert!((config.scale() - expected).abs() < 0.001);
    }

    #[test]
    fn test_attention_config_decoder_self() {
        let config = AttentionConfig::decoder_self_attention(8, 64, 100);
        assert!(config.causal);
        assert_eq!(config.seq_len_q, 100);
        assert_eq!(config.seq_len_kv, 100);
    }

    #[test]
    fn test_attention_config_cross() {
        let config = AttentionConfig::cross_attention(8, 64, 50, 1500);
        assert!(!config.causal);
        assert_eq!(config.seq_len_q, 50);
        assert_eq!(config.seq_len_kv, 1500);
    }

    #[test]
    fn test_gpu_attention_new() {
        let config = AttentionConfig::whisper_base();
        let attn = GpuAttention::new(config);
        assert_eq!(attn.config().num_heads, 8);
    }

    #[test]
    fn test_generate_scores_shader() {
        let config = AttentionConfig {
            num_heads: 4,
            head_dim: 32,
            seq_len_q: 64,
            seq_len_kv: 64,
            causal: false,
            tile_size: TileSize::Tile8x8,
        };
        let attn = GpuAttention::new(config);
        let shader = attn.generate_scores_shader();

        assert!(shader.contains("Attention Scores"));
        assert!(shader.contains("@compute"));
        assert!(shader.contains("SCALE"));
        assert!(shader.contains("tile_q"));
        assert!(shader.contains("tile_k"));
    }

    #[test]
    fn test_generate_causal_mask_shader() {
        let config = AttentionConfig::decoder_self_attention(8, 64, 100);
        let attn = GpuAttention::new(config);
        let shader = attn.generate_causal_mask_shader();

        assert!(shader.contains("Causal Mask"));
        assert!(shader.contains("NEG_INF"));
        assert!(shader.contains("col > row"));
    }

    #[test]
    fn test_generate_output_shader() {
        let config = AttentionConfig::whisper_tiny();
        let attn = GpuAttention::new(config);
        let shader = attn.generate_output_shader();

        assert!(shader.contains("Attention Output"));
        assert!(shader.contains("weights"));
        assert!(shader.contains("output"));
    }

    #[test]
    fn test_generate_fused_attention_shader() {
        let config = AttentionConfig::whisper_base();
        let attn = GpuAttention::new(config);
        let shader = attn.generate_fused_attention_shader();

        assert!(shader.contains("Fused Multi-Head Attention"));
        assert!(shader.contains("softmax"));
        assert!(shader.contains("SCALE"));
    }

    #[test]
    fn test_fused_attention_causal() {
        let config = AttentionConfig::decoder_self_attention(8, 64, 100);
        let attn = GpuAttention::new(config);
        let shader = attn.generate_fused_attention_shader();

        assert!(shader.contains("Causal mask"));
        assert!(shader.contains("col > row"));
    }

    #[test]
    fn test_fused_attention_non_causal() {
        let config = AttentionConfig::cross_attention(8, 64, 50, 1500);
        let attn = GpuAttention::new(config);
        let shader = attn.generate_fused_attention_shader();

        // Should not contain causal mask check
        assert!(!shader.contains("col > row"));
    }
}
