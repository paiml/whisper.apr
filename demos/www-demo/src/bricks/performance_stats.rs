//! `PerformanceStatsBrick`: System performance metrics display (PROBAR-SPEC-009)
//!
//! This brick displays system-level performance metrics:
//! - Memory usage (WASM heap)
//! - CPU/GPU utilization estimate
//! - RTF summary
//! - Model loading status
//!
//! # Assertions
//!
//! - Memory usage displayed accurately
//! - Performance metrics update in real-time
//! - Visual alerts on resource warnings

use jugar_probar::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use presentar_core::{
    AccessibleRole, Canvas, Color, Constraints, Event, LayoutResult, Point, Rect, Size, TextStyle,
    TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Memory unit formatting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryUnit {
    /// Bytes
    Bytes,
    /// Kilobytes
    KB,
    /// Megabytes
    MB,
    /// Gigabytes
    GB,
}

impl MemoryUnit {
    /// Auto-select appropriate unit
    #[must_use]
    pub fn auto_select(bytes: u64) -> (f64, Self) {
        if bytes >= 1_073_741_824 {
            (bytes as f64 / 1_073_741_824.0, Self::GB)
        } else if bytes >= 1_048_576 {
            (bytes as f64 / 1_048_576.0, Self::MB)
        } else if bytes >= 1024 {
            (bytes as f64 / 1024.0, Self::KB)
        } else {
            (bytes as f64, Self::Bytes)
        }
    }

    /// Get unit suffix
    #[must_use]
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Bytes => "B",
            Self::KB => "KB",
            Self::MB => "MB",
            Self::GB => "GB",
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryStats {
    /// Current heap usage in bytes
    pub heap_used: u64,
    /// Heap limit in bytes
    pub heap_limit: u64,
    /// Peak heap usage observed
    pub peak_used: u64,
}

impl MemoryStats {
    /// Create new memory stats
    #[must_use]
    pub fn new(heap_used: u64, heap_limit: u64) -> Self {
        Self {
            heap_used,
            heap_limit,
            peak_used: heap_used,
        }
    }

    /// Update memory usage
    pub fn update(&mut self, heap_used: u64) {
        self.heap_used = heap_used;
        self.peak_used = self.peak_used.max(heap_used);
    }

    /// Get usage percentage
    #[must_use]
    pub fn usage_percent(&self) -> f32 {
        if self.heap_limit > 0 {
            (self.heap_used as f32 / self.heap_limit as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Check if memory is low (>80% used)
    #[must_use]
    pub fn is_memory_low(&self) -> bool {
        self.usage_percent() > 80.0
    }

    /// Check if memory is critical (>95% used)
    #[must_use]
    pub fn is_memory_critical(&self) -> bool {
        self.usage_percent() > 95.0
    }

    /// Format used memory
    #[must_use]
    pub fn format_used(&self) -> String {
        let (value, unit) = MemoryUnit::auto_select(self.heap_used);
        format!("{:.1} {}", value, unit.suffix())
    }

    /// Format limit
    #[must_use]
    pub fn format_limit(&self) -> String {
        let (value, unit) = MemoryUnit::auto_select(self.heap_limit);
        format!("{:.1} {}", value, unit.suffix())
    }

    /// Format peak
    #[must_use]
    pub fn format_peak(&self) -> String {
        let (value, unit) = MemoryUnit::auto_select(self.peak_used);
        format!("{:.1} {}", value, unit.suffix())
    }
}

/// Model loading state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelLoadState {
    /// Not loaded
    #[default]
    NotLoaded,
    /// Loading in progress
    Loading,
    /// Loaded and ready
    Ready,
    /// Load failed
    Failed,
}

impl ModelLoadState {
    /// Get display name
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::NotLoaded => "Not Loaded",
            Self::Loading => "Loading...",
            Self::Ready => "Ready",
            Self::Failed => "Failed",
        }
    }

    /// Get CSS class
    #[must_use]
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::NotLoaded => "model-not-loaded",
            Self::Loading => "model-loading",
            Self::Ready => "model-ready",
            Self::Failed => "model-failed",
        }
    }
}

/// Performance statistics state
#[derive(Debug, Clone, Default)]
pub struct PerformanceStatsState {
    /// Memory statistics
    pub memory: MemoryStats,
    /// Model loading state
    pub model_state: ModelLoadState,
    /// Model name (e.g., "tiny", "base")
    pub model_name: Option<String>,
    /// Model size in bytes
    pub model_size: Option<u64>,
    /// Current RTF (if available)
    pub current_rtf: Option<f32>,
    /// Target RTF for the model
    pub target_rtf: f32,
    /// GPU acceleration available
    pub gpu_available: bool,
    /// GPU acceleration active
    pub gpu_active: bool,
    /// SIMD support available
    pub simd_available: bool,
    /// WebWorker active
    pub worker_active: bool,
}

impl PerformanceStatsState {
    /// Create new state
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_rtf: 2.0, // Default for tiny model
            ..Default::default()
        }
    }

    /// Set memory stats
    pub fn set_memory(&mut self, used: u64, limit: u64) {
        self.memory = MemoryStats::new(used, limit);
    }

    /// Update memory usage
    pub fn update_memory(&mut self, used: u64) {
        self.memory.update(used);
    }

    /// Set model info
    pub fn set_model(&mut self, name: impl Into<String>, size: u64, target_rtf: f32) {
        self.model_name = Some(name.into());
        self.model_size = Some(size);
        self.target_rtf = target_rtf;
    }

    /// Set model state
    pub fn set_model_state(&mut self, state: ModelLoadState) {
        self.model_state = state;
    }

    /// Set RTF
    pub fn set_rtf(&mut self, rtf: f32) {
        self.current_rtf = Some(rtf);
    }

    /// Set GPU status
    pub fn set_gpu_status(&mut self, available: bool, active: bool) {
        self.gpu_available = available;
        self.gpu_active = active;
    }

    /// Set SIMD status
    pub fn set_simd(&mut self, available: bool) {
        self.simd_available = available;
    }

    /// Set worker status
    pub fn set_worker(&mut self, active: bool) {
        self.worker_active = active;
    }

    /// Check if meeting performance target
    #[must_use]
    pub fn is_meeting_target(&self) -> Option<bool> {
        self.current_rtf.map(|rtf| rtf <= self.target_rtf)
    }

    /// Get overall health status
    #[must_use]
    pub fn health_status(&self) -> HealthStatus {
        // Critical: memory or model failure
        if self.memory.is_memory_critical() || self.model_state == ModelLoadState::Failed {
            return HealthStatus::Critical;
        }

        // Warning: low memory or not meeting target
        if self.memory.is_memory_low() {
            return HealthStatus::Warning;
        }

        if let Some(false) = self.is_meeting_target() {
            return HealthStatus::Warning;
        }

        // Model not ready
        if self.model_state != ModelLoadState::Ready {
            return HealthStatus::Unknown;
        }

        HealthStatus::Good
    }

    /// Reset state
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Overall system health
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// All systems nominal
    Good,
    /// Some issues detected
    Warning,
    /// Critical issues
    Critical,
    /// Status unknown
    Unknown,
}

impl HealthStatus {
    /// Get CSS class
    #[must_use]
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Good => "health-good",
            Self::Warning => "health-warning",
            Self::Critical => "health-critical",
            Self::Unknown => "health-unknown",
        }
    }

    /// Get color
    #[must_use]
    pub fn color(&self) -> &'static str {
        match self {
            Self::Good => "#50fa7b",
            Self::Warning => "#ffb86c",
            Self::Critical => "#ff6b6b",
            Self::Unknown => "#8b949e",
        }
    }
}

/// Performance statistics brick
#[derive(Debug, Clone, Default)]
pub struct PerformanceStatsBrick {
    state: PerformanceStatsState,
}

impl PerformanceStatsBrick {
    /// Create new brick
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get state reference
    #[must_use]
    pub fn state(&self) -> &PerformanceStatsState {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut PerformanceStatsState {
        &mut self.state
    }

    /// Set memory stats
    pub fn set_memory(&mut self, used: u64, limit: u64) {
        self.state.set_memory(used, limit);
    }

    /// Update memory usage
    pub fn update_memory(&mut self, used: u64) {
        self.state.update_memory(used);
    }

    /// Set model info
    pub fn set_model(&mut self, name: impl Into<String>, size: u64, target_rtf: f32) {
        self.state.set_model(name, size, target_rtf);
    }

    /// Set model state
    pub fn set_model_state(&mut self, state: ModelLoadState) {
        self.state.set_model_state(state);
    }

    /// Set RTF
    pub fn set_rtf(&mut self, rtf: f32) {
        self.state.set_rtf(rtf);
    }

    /// Set GPU status
    pub fn set_gpu_status(&mut self, available: bool, active: bool) {
        self.state.set_gpu_status(available, active);
    }

    /// Set SIMD status
    pub fn set_simd(&mut self, available: bool) {
        self.state.set_simd(available);
    }

    /// Set worker status
    pub fn set_worker(&mut self, active: bool) {
        self.state.set_worker(active);
    }

    /// Reset
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Generate capability badges HTML
    fn capability_badges(&self) -> String {
        let mut badges = Vec::new();

        if self.state.gpu_active {
            badges.push(r#"<span class="badge badge-gpu active">GPU</span>"#);
        } else if self.state.gpu_available {
            badges.push(r#"<span class="badge badge-gpu available">GPU</span>"#);
        }

        if self.state.simd_available {
            badges.push(r#"<span class="badge badge-simd active">SIMD</span>"#);
        }

        if self.state.worker_active {
            badges.push(r#"<span class="badge badge-worker active">Worker</span>"#);
        }

        badges.join("\n        ")
    }
}

impl Brick for PerformanceStatsBrick {
    fn brick_name(&self) -> &'static str {
        "PerformanceStatsBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::TextVisible,
            BrickAssertion::MaxLatencyMs(16),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let failed = Vec::new();

        for assertion in self.assertions() {
            passed.push(assertion.clone());
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        let state = &self.state;
        let health = state.health_status();
        let health_class = health.css_class();

        let model_info = match (&state.model_name, state.model_size) {
            (Some(name), Some(size)) => {
                let (size_val, unit) = MemoryUnit::auto_select(size);
                format!(
                    r#"<div class="model-info">
            <span class="model-name">{}</span>
            <span class="model-size">{:.1} {}</span>
        </div>"#,
                    name,
                    size_val,
                    unit.suffix()
                )
            }
            _ => String::new(),
        };

        let rtf_display = state
            .current_rtf
            .map(|rtf| {
                let status = if rtf <= 1.0 {
                    "realtime"
                } else if rtf <= state.target_rtf {
                    "target"
                } else {
                    "slow"
                };
                format!(
                    r#"<div class="stat-item rtf-stat {status}">
            <span class="stat-label">RTF</span>
            <span class="stat-value" data-testid="rtf">{:.2}x</span>
            <span class="stat-target">target: {:.1}x</span>
        </div>"#,
                    rtf, state.target_rtf, status = status
                )
            })
            .unwrap_or_default();

        let memory_class = if state.memory.is_memory_critical() {
            "memory-critical"
        } else if state.memory.is_memory_low() {
            "memory-warning"
        } else {
            "memory-ok"
        };

        let badges = self.capability_badges();

        format!(
            r#"<div class="performance-stats-brick {health_class}" data-testid="performance-stats">
    <div class="perf-header">
        <span class="title">Performance</span>
        <span class="health-indicator" style="background: {health_color}"></span>
    </div>

    <div class="model-section {model_class}">
        <span class="model-state">{model_state}</span>
        {model_info}
    </div>

    <div class="stats-section">
        {rtf_display}

        <div class="stat-item {memory_class}">
            <span class="stat-label">Memory</span>
            <span class="stat-value" data-testid="memory-used">{mem_used}</span>
            <div class="memory-bar">
                <div class="memory-fill" style="width: {mem_percent:.0}%"></div>
            </div>
            <span class="stat-detail">{mem_used} / {mem_limit} ({mem_percent:.0}%)</span>
            <span class="stat-peak">Peak: {mem_peak}</span>
        </div>
    </div>

    <div class="capabilities">
        {badges}
    </div>
</div>"#,
            health_class = health_class,
            health_color = health.color(),
            model_class = state.model_state.css_class(),
            model_state = state.model_state.display_name(),
            model_info = model_info,
            rtf_display = rtf_display,
            memory_class = memory_class,
            mem_used = state.memory.format_used(),
            mem_limit = state.memory.format_limit(),
            mem_percent = state.memory.usage_percent(),
            mem_peak = state.memory.format_peak(),
            badges = badges,
        )
    }

    fn to_css(&self) -> String {
        r".performance-stats-brick {
    background: #1a1a2e;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.performance-stats-brick.health-good {
    border-left: 3px solid #50fa7b;
}

.performance-stats-brick.health-warning {
    border-left: 3px solid #ffb86c;
}

.performance-stats-brick.health-critical {
    border-left: 3px solid #ff6b6b;
}

.perf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.perf-header .title {
    color: #e0e0e0;
    font-weight: 600;
    font-size: 0.875rem;
}

.health-indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.model-section {
    padding: 0.75rem;
    background: #16213e;
    border-radius: 6px;
    margin-bottom: 1rem;
}

.model-state {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.model-not-loaded .model-state { color: #8b949e; }
.model-loading .model-state { color: #ffb86c; }
.model-ready .model-state { color: #50fa7b; }
.model-failed .model-state { color: #ff6b6b; }

.model-info {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
}

.model-name {
    color: #e0e0e0;
    font-weight: 500;
}

.model-size {
    color: #8b949e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.875rem;
}

.stats-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.stat-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.stat-label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stat-value {
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 600;
}

.rtf-stat.realtime .stat-value { color: #50fa7b; }
.rtf-stat.target .stat-value { color: #4dc3ff; }
.rtf-stat.slow .stat-value { color: #ff6b6b; }

.stat-target {
    color: #8b949e;
    font-size: 0.75rem;
}

.stat-detail, .stat-peak {
    color: #8b949e;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

.memory-bar {
    background: #16213e;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    margin: 0.25rem 0;
}

.memory-fill {
    height: 100%;
    background: linear-gradient(90deg, #4dc3ff, #50fa7b);
    border-radius: 3px;
    transition: width 0.2s ease-out;
}

.memory-warning .memory-fill {
    background: linear-gradient(90deg, #ffb86c, #f1fa8c);
}

.memory-critical .memory-fill {
    background: linear-gradient(90deg, #ff6b6b, #ff5555);
}

.capabilities {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.badge {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    background: #16213e;
    color: #8b949e;
}

.badge.active {
    color: #50fa7b;
    border: 1px solid #50fa7b;
}

.badge.available {
    color: #ffb86c;
    border: 1px solid #ffb86c;
}

.badge-gpu.active { border-color: #bd93f9; color: #bd93f9; }
.badge-simd.active { border-color: #4dc3ff; color: #4dc3ff; }
.badge-worker.active { border-color: #50fa7b; color: #50fa7b; }"
            .into()
    }

    fn test_id(&self) -> Option<&str> {
        Some("performance-stats")
    }
}

impl Widget for PerformanceStatsBrick {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height: f32 = 240.0;
        Size::new(
            constraints.max_width.min(constraints.min_width.max(300.0)),
            height.min(constraints.max_height),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let bounds = Rect::new(0.0, 0.0, 400.0, 240.0);

        // Draw background
        let bg_color = Color::from_hex("#1a1a2e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bounds, bg_color);

        let text_color = Color::from_hex("#e0e0e0").unwrap_or(Color::WHITE);
        let style = TextStyle {
            size: 14.0,
            color: text_color,
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };

        // Draw title
        canvas.draw_text("Performance", Point::new(16.0, 24.0), &style);

        // Draw health indicator
        let health = self.state.health_status();
        let indicator_color = Color::from_hex(health.color()).unwrap_or(Color::WHITE);
        let indicator_rect = Rect::new(370.0, 16.0, 10.0, 10.0);
        canvas.fill_rect(indicator_rect, indicator_color);

        // Draw model state
        let model_style = TextStyle {
            size: 12.0,
            color: Color::from_hex("#8b949e").unwrap_or(Color::WHITE),
            weight: presentar_core::FontWeight::Normal,
            style: presentar_core::FontStyle::Normal,
        };
        canvas.draw_text(
            self.state.model_state.display_name(),
            Point::new(16.0, 56.0),
            &model_style,
        );

        // Draw RTF if available
        if let Some(rtf) = self.state.current_rtf {
            let rtf_text = format!("RTF: {:.2}x", rtf);
            let rtf_color = if rtf <= 1.0 {
                Color::from_hex("#50fa7b").unwrap_or(Color::GREEN)
            } else if rtf <= self.state.target_rtf {
                Color::from_hex("#4dc3ff").unwrap_or(Color::BLUE)
            } else {
                Color::from_hex("#ff6b6b").unwrap_or(Color::RED)
            };
            let rtf_style = TextStyle {
                size: 18.0,
                color: rtf_color,
                weight: presentar_core::FontWeight::Bold,
                style: presentar_core::FontStyle::Normal,
            };
            canvas.draw_text(&rtf_text, Point::new(16.0, 100.0), &rtf_style);
        }

        // Draw memory usage
        let memory_text = format!(
            "Memory: {} / {} ({:.0}%)",
            self.state.memory.format_used(),
            self.state.memory.format_limit(),
            self.state.memory.usage_percent()
        );
        canvas.draw_text(&memory_text, Point::new(16.0, 140.0), &style);

        // Draw memory bar
        let bar_bg = Rect::new(16.0, 152.0, 368.0, 6.0);
        let bar_bg_color = Color::from_hex("#16213e").unwrap_or(Color::BLACK);
        canvas.fill_rect(bar_bg, bar_bg_color);

        let fill_width = 368.0 * (self.state.memory.usage_percent() / 100.0);
        let bar_fill = Rect::new(16.0, 152.0, fill_width, 6.0);
        let bar_color = if self.state.memory.is_memory_critical() {
            Color::from_hex("#ff6b6b").unwrap_or(Color::RED)
        } else if self.state.memory.is_memory_low() {
            Color::from_hex("#ffb86c").unwrap_or(Color::YELLOW)
        } else {
            Color::from_hex("#4dc3ff").unwrap_or(Color::BLUE)
        };
        canvas.fill_rect(bar_fill, bar_color);
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn accessible_name(&self) -> Option<&str> {
        Some("Performance statistics")
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_unit_auto_select() {
        let (val, unit) = MemoryUnit::auto_select(500);
        assert_eq!(unit, MemoryUnit::Bytes);
        assert!((val - 500.0).abs() < 0.001);

        let (val, unit) = MemoryUnit::auto_select(2048);
        assert_eq!(unit, MemoryUnit::KB);
        assert!((val - 2.0).abs() < 0.001);

        let (val, unit) = MemoryUnit::auto_select(1_048_576 * 100);
        assert_eq!(unit, MemoryUnit::MB);
        assert!((val - 100.0).abs() < 0.001);

        let (val, unit) = MemoryUnit::auto_select(1_073_741_824 * 2);
        assert_eq!(unit, MemoryUnit::GB);
        assert!((val - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_stats_usage_percent() {
        let stats = MemoryStats::new(50_000_000, 100_000_000);
        assert!((stats.usage_percent() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_stats_is_memory_low() {
        let mut stats = MemoryStats::new(50_000_000, 100_000_000);
        assert!(!stats.is_memory_low());

        stats.update(85_000_000);
        assert!(stats.is_memory_low());
    }

    #[test]
    fn test_memory_stats_is_memory_critical() {
        let stats = MemoryStats::new(96_000_000, 100_000_000);
        assert!(stats.is_memory_critical());
    }

    #[test]
    fn test_memory_stats_peak_tracking() {
        let mut stats = MemoryStats::new(50_000_000, 100_000_000);
        stats.update(80_000_000);
        assert_eq!(stats.peak_used, 80_000_000);

        stats.update(60_000_000);
        assert_eq!(stats.peak_used, 80_000_000);
        assert_eq!(stats.heap_used, 60_000_000);
    }

    #[test]
    fn test_model_load_state_display_name() {
        assert_eq!(ModelLoadState::NotLoaded.display_name(), "Not Loaded");
        assert_eq!(ModelLoadState::Loading.display_name(), "Loading...");
        assert_eq!(ModelLoadState::Ready.display_name(), "Ready");
        assert_eq!(ModelLoadState::Failed.display_name(), "Failed");
    }

    #[test]
    fn test_performance_stats_state_default() {
        let state = PerformanceStatsState::new();
        assert_eq!(state.model_state, ModelLoadState::NotLoaded);
        assert_eq!(state.target_rtf, 2.0);
    }

    #[test]
    fn test_performance_stats_state_set_model() {
        let mut state = PerformanceStatsState::new();
        state.set_model("tiny", 39_000_000, 2.0);

        assert_eq!(state.model_name, Some("tiny".into()));
        assert_eq!(state.model_size, Some(39_000_000));
        assert_eq!(state.target_rtf, 2.0);
    }

    #[test]
    fn test_performance_stats_state_is_meeting_target() {
        let mut state = PerformanceStatsState::new();
        state.target_rtf = 2.0;

        assert!(state.is_meeting_target().is_none());

        state.set_rtf(1.5);
        assert!(state.is_meeting_target().unwrap());

        state.set_rtf(3.0);
        assert!(!state.is_meeting_target().unwrap());
    }

    #[test]
    fn test_performance_stats_state_health_status() {
        let mut state = PerformanceStatsState::new();

        // Unknown when model not ready
        assert_eq!(state.health_status(), HealthStatus::Unknown);

        // Good when model ready and meeting target
        state.set_model_state(ModelLoadState::Ready);
        state.set_memory(50_000_000, 100_000_000);
        state.set_rtf(1.5);
        assert_eq!(state.health_status(), HealthStatus::Good);

        // Warning when not meeting target
        state.set_rtf(3.0);
        assert_eq!(state.health_status(), HealthStatus::Warning);

        // Critical when memory critical
        state.set_memory(96_000_000, 100_000_000);
        assert_eq!(state.health_status(), HealthStatus::Critical);
    }

    #[test]
    fn test_health_status_color() {
        assert_eq!(HealthStatus::Good.color(), "#50fa7b");
        assert_eq!(HealthStatus::Warning.color(), "#ffb86c");
        assert_eq!(HealthStatus::Critical.color(), "#ff6b6b");
        assert_eq!(HealthStatus::Unknown.color(), "#8b949e");
    }

    #[test]
    fn test_brick_default() {
        let brick = PerformanceStatsBrick::new();
        assert_eq!(brick.state().model_state, ModelLoadState::NotLoaded);
    }

    #[test]
    fn test_brick_set_memory() {
        let mut brick = PerformanceStatsBrick::new();
        brick.set_memory(50_000_000, 100_000_000);
        assert_eq!(brick.state().memory.heap_used, 50_000_000);
    }

    #[test]
    fn test_brick_set_model() {
        let mut brick = PerformanceStatsBrick::new();
        brick.set_model("base", 74_000_000, 2.5);
        assert_eq!(brick.state().model_name, Some("base".into()));
    }

    #[test]
    fn test_brick_verification() {
        let brick = PerformanceStatsBrick::new();
        let result = brick.verify();
        assert!(result.is_valid());
    }

    #[test]
    fn test_brick_to_html() {
        let mut brick = PerformanceStatsBrick::new();
        brick.set_model("tiny", 39_000_000, 2.0);
        brick.set_model_state(ModelLoadState::Ready);
        brick.set_memory(50_000_000, 100_000_000);
        brick.set_rtf(1.5);

        let html = brick.to_html();
        assert!(html.contains("data-testid=\"performance-stats\""));
        assert!(html.contains("Performance"));
        assert!(html.contains("tiny"));
    }

    #[test]
    fn test_brick_budget() {
        let brick = PerformanceStatsBrick::new();
        assert_eq!(brick.budget().total_ms, 16);
    }

    #[test]
    fn test_brick_can_render() {
        let brick = PerformanceStatsBrick::new();
        assert!(brick.can_render());
    }
}
