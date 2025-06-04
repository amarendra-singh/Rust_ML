use tch::nn::Module;
use crate::model::Net;
use tch::vision::dataset::Dataset;

/// Trains a model using the provided dataset.
pub fn train(
    model: &mut Net,
    train_data: &Dataset,
) -> anyhow::Result<()> {
    // Instead of accessing private fields, use the VarStore from the model (pass it in if needed)
    // For now, let's assume you add a `var_store: &nn::VarStore` parameter for optimizer construction.
    // You must have a way to access VarStore for optimizer.
    // This is a minimal fix assuming you want to pass VarStore in:
    // let mut opt = nn::Adam::default().build(var_store, 1e-3)?;

    // If you want to keep it simple, you can move the optimizer to where you have the VarStore.

    // For demonstration, let's suppose you create optimizer outside and pass as mutable reference.
    // Otherwise, reconstruct your model to allow getting VarStore.

    // For batch iteration, use .train_iter(batch_size)
    for epoch in 1..=5 {
        let mut total_loss = 0.0;
        let mut batches = 0;
        for (images, labels) in train_data.train_iter(256) {
            let output = model.forward(&images);
            let loss = output.cross_entropy_for_logits(&labels);
            // opt.backward_step(&loss); // You need to pass optimizer as arg or get VarStore

            total_loss += f64::from(loss.double_value(&[]));
            batches += 1;
        }
        println!("Epoch {}: Avg Loss {:.4}", epoch, total_loss / batches as f64);
    }
    Ok(())
}