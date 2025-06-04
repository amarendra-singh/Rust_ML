use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

#[derive(Debug)] // Needed for tch::nn::Module
pub struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    pub fn new(vs: &nn::Path) -> Self {
        let fc1 = nn::linear(vs, 784, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default());
        Self { fc1, fc2 }
    }
}

impl nn::Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.view([-1, 784]).apply(&self.fc1).relu().apply(&self.fc2)
    }
}

#[derive(Debug)]
pub struct MnistModel {
    vs: nn::VarStore,
    net: Net,
}

impl MnistModel {
    pub fn new() -> Self {
        let vs = nn::VarStore::new(Device::cuda_if_available());
        let net = Net::new(&vs.root());
        Self { vs, net }
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut model = Self::new();
        model.vs.load(path)?;
        Ok(model)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }

    pub fn train(&mut self) -> Result<()> {
        let mnist = tch::vision::mnist::load_dir("data")?;
        let mut opt = nn::Adam::default().build(&self.vs, 1e-3)?;
        for epoch in 1..=5 {
            let mut total_loss = 0.0;
            let mut batches = 0;
            for (images, labels) in mnist.train_iter(256) {
                let loss = self.net.forward(&images).cross_entropy_for_logits(&labels);
                opt.backward_step(&loss);
                total_loss += f64::from(loss.double_value(&[]));
                batches += 1;
            }
            println!("Epoch {}: Loss: {:.4}", epoch, total_loss / batches as f64);
        }
        Ok(())
    }

    pub fn predict(&self, pixels: &[f32]) -> i64 {
        // Use f_from_slice to construct a Tensor from &[f32]
        let tensor = Tensor::f_from_slice(pixels).unwrap().view([1, 784]);
        self.net.forward(&tensor).argmax(1, false).int64_value(&[0])
    }
}