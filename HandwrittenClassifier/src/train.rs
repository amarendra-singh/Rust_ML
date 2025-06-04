use tch::nn::Module;
use crate::model::Net;
use tch::vision::dataset::Dataset;

/// Trains a model using the provided dataset.
pub fn train(
    model: &mut Net,
    train_data: &Dataset,
) -> anyhow::Result<()> {

    for epoch in 1..=5 {
        let mut total_loss = 0.0;
        let mut batches = 0;
        for (images, labels) in train_data.train_iter(256) {
            let output = model.forward(&images);
            let loss = output.cross_entropy_for_logits(&labels);

            total_loss += f64::from(loss.double_value(&[]));
            batches += 1;
        }
        println!("Epoch {}: Avg Loss {:.4}", epoch, total_loss / batches as f64);
    }
    Ok(())
}