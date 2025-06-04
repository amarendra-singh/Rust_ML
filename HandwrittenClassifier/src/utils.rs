use anyhow::Result;

use crate::model::MnistModel;

/// Loads an image and preprocesses it for MNIST model inference.
pub fn load_and_preprocess_image(path: &str) -> Result<Vec<f32>> {
    let img = image::open(path)?;
    let img = img.resize_exact(28, 28, image::imageops::FilterType::Lanczos3).to_luma8();
    Ok(img.pixels().map(|p| p.0[0] as f32 / 255.0).collect())
}

/// Predicts a digit from an image file using a trained model.
pub fn predict_from_image(image_path: &str, model_path: &str) -> Result<i64> {
    let model = MnistModel::load(model_path)?;
    let pixels = load_and_preprocess_image(image_path)?;
    Ok(model.predict(&pixels))
}