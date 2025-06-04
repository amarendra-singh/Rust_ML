mod data;
mod model;
mod train;
mod utils;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    Train,
    Predict {
        image: String,
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