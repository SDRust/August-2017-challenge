# August-2017-challenge

From the TensorFlow web page:
```
TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API.
```

Recently, TensorFlow started exposing a C API which was quickly used to implement Rust bindings.
One shortcoming of the current C API is it does not expose an easy way to construct graphs (only to
run them). Therefore, this demo uses these bindings to load in a serialized network, feed it some
data, and look at the output.

The network used here is the "hello world" of deep learning, MNIST MLP described [in this bit of the TensorFlow
documentation](https://www.tensorflow.org/get_started/mnist/beginners).

## How to use this example
(Optional: since I committed the serialized model, you don't need to do this yourself now)
- Install TensorFlow for running the python script.
- Run `./run.sh`

- `cargo run`

## Task ideas

Straightforward:
- Run a single digit through the classifier and get the predicted probabilities for each digit
    (class).
- Test the average accuracy of the classifier using the testing set, visualize digits the classifier
    gets wrong.
- Find and visualize the digit in each class that best represents that class (has the highest
    correct predicted probability).

Harder:
- Load digits from images on disk, rescale and feed into the classifier.
- Train the model with rust instead of python.
