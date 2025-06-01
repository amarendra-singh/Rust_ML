mod data;
mod model;
mod train;

use anyhow:: Result;
use clap::Command;
use mnist_classifier::{MnistModel, predict_from_image};

use model::Net;
use tch::vision::{image, mnist};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(subcommand)]
enum Commands {
    Train,
    Predict {
        image: Stirng,
        model: String
    }
}


fn main() -> Result<()> {
    let cli = Cli::parser();

    match cli.command {
        Commands::Train => {
            println!("Training model..");
            let model = MnistModel::new();
            model.train()?;
            model.save("model.pt")?;
            println!("Model saved to model.pt");

        }

        Commands::Predict { image, model } => {
            println!("Predicting digit from {}..", image);
            let digit = predict_from_image(&image, &model)?;
            println!("Predicted digit{}", digit);
        }
    }

    Ok(())
}


// fn main() -> Result<()> {

//     let (train_data, test_data) = data::load_data()?;

//     let vs = tch::nn::VarStore::new(tch::Device::cuda_if_available());
//     let mut model = Net::new(&vs.root());

//     train::train(&mut model, &train_data)?;

//     let accuracy = test_model(&model, &test_data)?;

//     println!("Test accuracy: {:.2}%", accuracy *100);

//     Ok(())
// }


// fn test_model(model: &Net, test_data: &mnist::Dataset) -> Result<f64> {
//     let mut correct = 0;
//     for (image, labels) in test_data.iter(1024) {
//         let prediction = model.forward(&image).argmax(1, false);
//         correct += prediction.eq(&labels).sum().int64_value(&[]);

//     }
//     ok(correct as f64 / test_data.labels.size()[0] as f64)
// }
