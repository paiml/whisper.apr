//! Tensor statistics for validation

/// Statistics for a tensor
#[derive(Debug, Clone, PartialEq)]
pub struct TensorStats {
    /// Tensor name
    pub name: String,
    /// Number of elements
    pub count: usize,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of Inf values
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
}

impl TensorStats {
    /// Compute statistics for a tensor
    pub fn compute(name: &str, data: &[f32]) -> Self {
        let count = data.len();
        if count == 0 {
            return Self {
                name: name.to_string(),
                count: 0,
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
            };
        }

        let mut sum = 0.0_f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                sum += v as f64;
                if v == 0.0 {
                    zero_count += 1;
                }
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };

        // Compute std
        let mut var_sum = 0.0_f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = (v as f64) - (mean as f64);
                var_sum += diff * diff;
            }
        }
        let std = if valid_count > 1 {
            ((var_sum / valid_count as f64).sqrt()) as f32
        } else {
            0.0
        };

        Self {
            name: name.to_string(),
            count,
            mean,
            std,
            min,
            max,
            nan_count,
            inf_count,
            zero_count,
        }
    }

    /// Check if tensor is all zeros
    #[must_use]
    pub fn is_all_zeros(&self) -> bool {
        self.zero_count == self.count
    }

    /// Check if tensor has any NaN values
    #[must_use]
    pub fn has_nan(&self) -> bool {
        self.nan_count > 0
    }

    /// Check if tensor has any Inf values
    #[must_use]
    pub fn has_inf(&self) -> bool {
        self.inf_count > 0
    }
}
