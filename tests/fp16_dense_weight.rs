//! whisper.apr#103: the fp16 path frees `LinearWeights::weight`, so GPU uploads
//! need a dense f32 view that works for either storage.
use whisper_apr::model::LinearWeights;

#[test]
fn dense_weight_survives_fp16_conversion() {
    let mut w = LinearWeights::new(5, 3);
    for (i, v) in w.weight.iter_mut().enumerate() {
        *v = i as f32 * 0.5;
    }
    let want = w.weight.clone();
    w.convert_to_f16();
    assert!(w.weight.is_empty(), "fp16 path frees f32 weights");
    assert_eq!(w.dense_weight().as_ref(), want.as_slice());
}
