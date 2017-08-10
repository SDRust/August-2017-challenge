#![cfg_attr(feature="nightly", feature(alloc_system))]
#[cfg(feature="nightly")]
extern crate alloc_system;
extern crate tensorflow;
extern crate curl;
extern crate mnist;
extern crate rulinalg;

use std::io::Write;
use std::fs::{File, create_dir};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::error::Error;

use curl::easy::Easy;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;
use mnist::{Mnist, MnistBuilder};
use rulinalg::matrix::{BaseMatrix, Matrix};

fn main() {
    if !Path::new("data").exists() {
        create_dir("data").expect("Could not create 'data' directory");
    }

    let img_urls = &["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                     "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                     "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                     "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"];

    for img_url in img_urls {
        let mut easy = Easy::new();
        // My bleeding eyes!!
        let fname: PathBuf = ["data", Path::new(img_url).file_name().unwrap().to_str().unwrap()].iter().collect();
        let unpacked_name = fname.with_extension("");
        if !unpacked_name.exists() && !fname.exists() {
            // Download packed file
            let mut fid = File::create(&fname).unwrap();
            println!("Downloading {:?}", &fname);
            easy.url(img_url).unwrap();
            easy.write_function(move |data| {
                Ok(fid.write(data).unwrap())
            }).unwrap();
            easy.perform().unwrap();
        }
        if !unpacked_name.exists() {
            // Unpack data files
            Command::new("gunzip")
                .arg("-k")
                .arg(&fname.file_name().unwrap())
                .current_dir(fname.parent().unwrap())
                .spawn()
                .expect(&format!("Could not unpack {:?}", fname.clone()));
        }
    }

    match run() {
        Err(e) => println!("Error: {}", e.to_string()),
        Ok(_) => ()
    }
}


fn run() -> Result<(), Box<Error>> {
    let export_dir = "./saved_model";

    use std::env;
    let path = env::current_dir().unwrap();
    println!("The current directory is {}", path.display());

    if !Path::new(export_dir).exists() {
        return Err(Box::new(Status::new_set(Code::NotFound,
                                &format!("Could not find the folder {}, run 'run.sh' to generate it.", export_dir))
            .unwrap()));
    }

    let (trn_size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .base_path("data")
        .training_set_length(trn_size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    println!("Loaded mnist data");

    {
        // Get the label of the first digit.
        let first_label = trn_lbl[0];
        println!("The first digit is a {}.", first_label);

        // Convert the flattened training images vector to a matrix.
        let trn_img = Matrix::new((trn_size * rows) as usize, cols as usize, trn_img.clone());

        // Get the image of the first digit.
        let row_indexes = (0..27).collect::<Vec<_>>();
        let first_image = trn_img.select_rows(&row_indexes);
        println!("The image looks like... \n{}", first_image);
    }

    // Generate an input test image.

    // TODO: Allocate TensorFlow `Tensor`s for input and output and fill them with data for a
    // single image

    // Load the saved model exported by regression_savedmodel.py.
    let mut graph = Graph::new();
    let mut session = Session::from_saved_model(&SessionOptions::new(), 
                                                &["train", "serve"],
                                                &mut graph,
                                                export_dir)?;
    // TODO: Pull out the input (`x`) and output (`prediction`) ops by name from the graph


    // Make a single prediction
    let mut prediction_step = StepWithGraph::new();

    // TODO: 
    // Add x as an input (binding it to data) 
    // Request the prediction output) 
    // then run the session.
    // Then extract the results and print

    Ok(())
}
