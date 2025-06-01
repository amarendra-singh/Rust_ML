use tch::{nn, nn::OptimizerConfig, Device};

pub fn train(
    model: &mut Net,
    train_data: &tch::vision::mnist::Dataset ) ->
    anyhow::Result<()>{
        let mut opt = nn::Adman::default().build(&model.parameters(), 1e-3)?;

        for epoch in 1..5{
            let mut total_loss = 0.0;
            let mut batches = 0;

            for (images, labels) in train_data.iter(256).shuffle(){
                let output = model.forward(&images);
                let loss = output.cross_entropy_for_logits(&label);

                opt.backward_step(&loss);

                total_loss += f64::form(loss);
                batches +=1;
            }

            println!("Epoch {}: Avg Loss {:.4}", epoch, total_loss / batches as f64);
            
        }
        ok(())
    }