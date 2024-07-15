## Files explanation
* **XXX_data.npy**: numpy data file
* **plot_data.ipynb**: import the numpy file and plot it
* **XXX_modelState_(Testing_accuracy)**: network theta parameters files
* **QCNN_3D_XXX.py**: import the modelState file and train this network, usually you can just run it

## Hyperparameters
Most hyperparameters are same and constant, but we need to change a little during training:

* Learning_rate=1e-2 for the first 10 epochs with 0.95 decay
* Learning_rate=1e-3 for the second 10 epochs with 0.95 decay
* learning_rate=1e-4 for the third 10 epochs without decay (if it has)

except:
* channel_number=7, strid=2 for MNIST/FashionMNIST, but channel_number=6, stride=1 for CIFAR-10
* output_scale=30 for MNIST/CIFAR-10, but output_scale=20 for FashionMNIST