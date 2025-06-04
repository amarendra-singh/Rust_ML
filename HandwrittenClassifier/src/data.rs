use tch::vision::{mnist, dataset::Dataset};

pub fn load_data() -> anyhow::Result<Dataset> {
    let mnist = mnist::load_dir("data")?;
    Ok(mnist)
}