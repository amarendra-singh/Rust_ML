use anyhow::Ok;
use image::Pixel;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Failed to load data")]
    DataLoadError(#[From] tech::TchError),
    #[error("IO error")]
    IOError(#[from] std::io::Error),
}

pub struct MnistModel {
    vs: nn::VarStore,
    net: Net,
}

pub struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}


impl Net {
    pub fn new(vs: &nn::Path) -> Self{
        let fc1 = nn::linear(vs, 784,128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());
        self {fc1, fc2}
    }
}

impl Module for Net{
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 784]).apply(&self.fc1).relu().apply(&self.fc2)
    }
}

impl MnistModel{
    pub fn new() -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let net = Net::new(&vs.root());
        Self {vs, net}
    }

    pub fn load(path: &str) -> Result<(Self, ModelError)> {
        let model = Self::new();
        model.vs.load(path)?;
        Ok(model)
    }

    pub fn save(&self, path: &str) -> Result<(), ModelError>{
        self.vs.save(path)?;
        Ok(())
    }

    pub fn train (&self) -> Result<(), ModelError>{
        let mnist = tch::vision::mnist::load_dir("data")?;
        let mut opt = nn::Adam::default().build(&self.vs.variables(), 1e-3)?;

        for epoch in 1..=5 {
            let mut total_loss = 0.0;
            let mut batches = 0;

            for (image, labels) in mnist.train.iter(256).shuffle() {
                let loss = self.net.forward(&image).cross_entropy_for_logits(&labels);
                opt.backward_step(&loss);
                total_loss += f64::from(loss);
                batches += 1;
            }

            println!("Epoch {}: Loss: {:.4}", epoch, total_loss / batches as f64);
        }
        Ok(())
    }

    pub fn predict(&self, pixels: &[f32]) -> i64 {
        let tensor = Tensor::of_slice(pixels).view ([1, 784]);
        self.net.forward(&tensor).argmax(1, false).int64_value(&[0])
    }
}