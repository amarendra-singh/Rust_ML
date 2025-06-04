use tch::vision::{mnist, dataset::Dataset};

/// Loads the MNIST dataset from the `data` folder.
pub fn load_data() -> anyhow::Result<Dataset> {
    // The Dataset type is public in tch::vision::dataset
    let mnist = mnist::load_dir("data")?;
    Ok(mnist)
}