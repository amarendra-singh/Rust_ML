use tch::{vision""mnist, Device};

pub fn load_data() -> anyhow::Result<(mnist::Dataset, mnist::Dataset)>{
    let mnist = mnist::load_dir("data)?;
    ok ((mnist.train, mnist.test))
}