# Models comparison tool &nbsp; [![Generic badge](https://img.shields.io/static/v1?style=social&logo=youtube&label=YouTube&message=Watch%20now!&url=www.youtube.com/c/symbolxchannel)](https://www.youtube.com/user/symbolxchannel/playlists)
> René Chenard - June 2020

Generates animated plots of different models of neural network classifiers during learning in order to compare them.

| ![Tanh networks](/Amounts%20of%20neurons/amount_of_neurons_tanh_sequential.gif?raw=true "Tanh networks with different amounts of neurons per layer (Sequential, not shuffled).") |
|:--:|
| Figure 1 &nbsp;&mdash;&nbsp; Tanh networks with different amounts of neurons per layer (Sequential, not shuffled). |

| ![Tanh networks](/Amounts%20of%20neurons/amount_of_neurons_tanh_batch.gif?raw=true "Tanh networks with different amounts of neurons per layer (Batches of 32, shuffled).") |
|:--:|
| Figure 2 &nbsp;&mdash;&nbsp; Tanh networks with different amounts of neurons per layer (Batches of 32, shuffled). |

---

### Content:

Directory | Description | Links
--- | --- | :---:
Activation functions | Comparison of the effect of using different types of activation function. | &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Activation%20functions/activation_functions1.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Activation%20functions/activation_functions2.ipynb) &nbsp;
Amounts of layers | Comparison of the effect of using different depths of layers. | &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Amounts%20of%20layers/amount_of_layers_relu.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Amounts%20of%20layers/amount_of_layers_tanh.ipynb) &nbsp;
Amounts of neurons | Comparison of the effect of using different amounts of neurons per layer. | &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Amounts%20of%20neurons/amount_of_neurons_relu.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Amounts%20of%20neurons/amount_of_neurons_tanh.ipynb) &nbsp;
Learning rate | Comparison of the effect of using different learning rates. | &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Learning%20rate/learning_rate_relu.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Learning%20rate/learning_rate_tanh.ipynb) &nbsp;
Loss functions | Comparison of the effect of using different loss functions. | &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Loss%20functions/loss_functions_relu.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Loss%20functions/loss_functions_tanh.ipynb) &nbsp;
Optimizers | Comparison of the effect of using different optimizers. |  &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Optimizers/optimizers_relu1.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Optimizers/optimizers_relu2.ipynb) &emsp; [&#9314;](https://github.com/RECHE23/models_comparison/blob/master/Optimizers/optimizers_tanh1.ipynb) &emsp; [&#9315;](https://github.com/RECHE23/models_comparison/blob/master/Optimizers/optimizers_tanh2.ipynb) &nbsp;
Weight decay | Comparison of the effect of using different values of weight decay. | &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Weight%20decay/weight_decay_relu.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Weight%20decay/weight_decay_tanh.ipynb) &nbsp;
Weight initialization | Comparison of the effect of using different weight initialization methods. | &nbsp; [&#9312;](https://github.com/RECHE23/models_comparison/blob/master/Weight%20initialization/weight_initialization_relu.ipynb) &emsp; [&#9313;](https://github.com/RECHE23/models_comparison/blob/master/Weight%20initialization/weight_initialization_tanh.ipynb) &nbsp;

---

### Details:
For simplicity reasons, these experiments are done on simple neural network classifiers with two classes. A negative output corresponds to one class while a positive output corresponds to the other class. The [error function](https://en.wikipedia.org/wiki/Error_function) is applied on the output neuron to ensure that the output is always between -1 and 1.

Unless specified, the neural networks are composed of four layers (two hidden layers) with 32 neurons per layer and uses the [stochastic gradient descent optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) with a [learning rate](https://en.wikipedia.org/wiki/Learning_rate) of 0.01, a [momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) of 0.9, [weight decay](https://en.wikipedia.org/wiki/Regularization_(mathematics)#Tikhonov_regularization) at 0.001. The default [loss function](https://en.wikipedia.org/wiki/Loss_functions_for_classification) is the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error).

The sampling methods used by default are:
- sequential learning over the whole dataset (gradient is calculated after feeding the whole dataset in the same sequential order)
- Shuffled batches of 32 samples

<br />

| :warning: CAUTION!      |
| :--------------------------- |
| **The results obtained by these experiments may vastly diverge with slightly different parameters.<br />Always verify what you infer from them!** |

---

### TO DO:
- [ ] Explore alternative learning methods.
- [ ] Use one frame per retropopagation instead of one frame per epoch, for smoother animations.
- [ ] Implement [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) (like [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), [UMAP](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection) or [tSNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)) to analyze higher dimensions datasets.
- [ ] Display more information about the setup: loss function, architecture, optimizer.
- [ ] Display more information about the learning: learning curve, weights & biases, metrics, etc.
- [ ] Make a module or an application out of the Jupyter Notebook experiments.
- [ ] Add different color themes.
- [ ] Add support for 0-1 classifiers and multiple classes classifiers.
