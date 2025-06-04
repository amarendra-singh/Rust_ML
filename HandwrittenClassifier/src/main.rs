mod data;
mod model;
mod train;
mod utils;

use anyhow::Result;
use clap::Parser;

/// Command-line interface for the handwritten digit classifier.
#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Available subcommands for the CLI.
#[derive(clap::Subcommand)]
enum Commands {
    /// Train the model on the MNIST dataset.
    Train,
    /// Predict the digit in a given image using a trained model.
    Predict {
        /// Path to the image file.
        image: String,
        /// Path to the model file.
        model: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train => {
            println!("Training model...");
            let mut model = model::MnistModel::new();
            model.train()?;
            model.save("model.pt")?;
            println!("Model saved to model.pt");
        }
        Commands::Predict { image, model } => {
            println!("Predicting digit from {}...", image);
            let digit = utils::predict_from_image(&image, &model)?;
            println!("Predicted digit: {}", digit);
        }
    }

    Ok(())
}