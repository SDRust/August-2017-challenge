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
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

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
        Ok(s) => println!("")
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
    let mut x = Tensor::new(&[1, (rows*cols) as u64]);
    let mut y = Tensor::new(&[10 as u64]);
    y[trn_lbl[0] as usize] = 1.0;
    for i in 0..28*28 {
        x[i] = trn_img[i] as f32 / 255.0;
    }

    // Load the saved model exported by regression_savedmodel.py.
    let mut graph = Graph::new();
    let mut session = Session::from_saved_model(&SessionOptions::new(), 
                                                &["train", "serve"],
                                                &mut graph,
                                                export_dir)?;
    let op_x = graph.operation_by_name_required("x")?;
    // let op_y = graph.operation_by_name_required("y")?;
    let op_prediction = graph.operation_by_name_required("prediction")?;
    // let op_train = graph.operation_by_name_required("train")?;

    // Train the model (e.g. for fine tuning).
    // let mut train_step = StepWithGraph::new();
    // train_step.add_input(&op_x, 0, &x);
    // train_step.add_input(&op_y, 0, &y);
    // train_step.add_target(&op_train);
    // for _ in 0..steps {
    //     session.run(&mut train_step)?;
    // }

    // Make a single prediction
    let mut prediction_step = StepWithGraph::new();
    prediction_step.add_input(&op_x, 0, &x);
    prediction_step.add_target(&op_prediction);
    let y_ix = prediction_step.request_output(&op_prediction, 0);
    session.run(&mut prediction_step)?;
    println!("y_ix: {:?} output_type: {:?}", y_ix, prediction_step.output_data_type(0));

    // Check our results.
    let y_val: &[f32] = &prediction_step.take_output(y_ix)?;
    println!("{:?}", y_val);
    Ok(())
}
