# Adversarial Machine Learning on a Traffic Sign recognition system

This is a repository which contains the implementation of different Adversarial Machine Learning black-box attacks in order to check the risk that they would related to a traffic sign recognition system.

For this experimentation, it has been created an implementation of the neural network defined by [Mishra and Goyal](https://link.springer.com/article/10.1007/s11042-022-12531-w), as model on which conduct the attacks.<br/>

[Mishra and Goyal](https://link.springer.com/article/10.1007/s11042-022-12531-w) defined a convolutional neural network (CNN), which is able to classify a traffic sign as one of the 43 possible labels.

During this experimentation, several attacks have been conducted:
- Zeroth-order optimization attack (ZOO)<br/>
A black-box evasion attack, implemented in it's untargeted and targeted version.<br/>
This attack occurs during the deployment stage, and so after the model's train. This attack plans to take some test examples and applies a small perturbation on them, resulting in adversarial examples.<br/>
Adversarial examples created are so examples which differ a little from the original ones, but are misclassified by the model (so they are classified as a label different from the original).<br/>
[Link to the paper](https://arxiv.org/abs/1708.03999)
- BadNets<br/>
A back-box backdoor poisoning attack which occurs during the training stage and plans to get a some examples of the training dataset and apply a backdoor trigger on them, resulting in the poisoned examples.<br/>
Poisoned examples are so just the same of original training examples but with a trigger on them.<br/>
These poisoned examples are finally added to the training dataset with which the model trains.<br/>
[Link to the paper](https://arxiv.org/abs/1708.06733)
- Clean-label Black-Box backdoor poisoning attack<br/>
A White-box backdoor poisoning attack inspired by BadNets but applied in a clean-label scenario. This attack plans to perturb selected examples by using an evasion attack (in this case PGD attack), resulting in adversarial examples to which is applied the backdoor trigger, resulting in the poisoned samples which are replaced to original samples in the training dataset.<br/>
In this experimentation, the attack has been implemented as black-box attack by using ZOO attack as evasion attack.<br/>
[Link to the paper](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf)

In order to optimize all these attack, the following tecniques have been used:
- Bayesian Optimization<br/>
An optimization method which plans to approximate input parameters of a function such that they maximize the function itself. It requires the space value in which parameters are.<br/>
This had been used in ZOO attack in order to optimize attack's parameters to reach better results.
- Grid Search<br/>
An optimization method which plans to try all possible combination of parameters' values, inputs to a function, in order to find the combinaion which maximize the function itself.<br/>
This had been used in BadNets attack in order to optimize attack's parameters to reach better results.

## The Model

The used model, takes an image in form of numpy array of shape (32,32,3), and output a vector of 43 elements; the _i-th_ element of the output, represent the probability that the image belongs to the _i-th_ class of sign.

Here is represented the model's architecture

![alt text](https://github.com/mattiacolucci/Adversarial-Machine-Learning-On-Traffic-Sign-Recognition/blob/main/Images/model_architecture.png?raw=true "Model's architecture")
> Image took from the [paper of the model](https://link.springer.com/article/10.1007/s11042-022-12531-w)

The model has been trained of the [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). This dataset is composed by 39209 german traffi signs. In order to import it you can download the dataset from Kaggle and put the ```Train``` folder into a folder called ```Dataset``` in the root of this repository.

Our implementation of the model has achived these performances:
| Accuracy      | Loss    | Avg Precision  | Avg Recall |
| ------------- |:-------:|:--------------:|:----------:|
| 99.3%         | 0.022   | 99.8%          | 99.7%      |
| 99.2%         | 0.029   | 99.3%          | 98.9%      |

## Packages used

In this experimentation the following packages/libraries have been used:
- Adversarial-Robustness-Toolbox (ART)<br/>
A python package which provides several attacks. BadNets attack has been implemented using this library.<br/>
[Link to the library](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- Autozoom Attack<br/>
A library which implements ZOO and AutoZOOM attack by [Chen](https://arxiv.org/abs/1805.11770). ZOO attack has been implemented using the ZOO implementation provied into this library, which code is in the ```Autozoomattack``` folder of this repository.<br/>
[Link to the library](https://github.com/IBM/Autozoom-Attack)
- BayesOpt
A python library which implements bayesian optimization.
[Link to the library](https://bayesian-optimization.github.io/BayesianOptimization/quickstart.html)

In order to use these libraries, you just need to follow the steps explained in the [Installation](#installation) part.

## Installation with Conda

This experimentation has been conducted using Python 3.6.10 and Tensorflow 1.6.0.

In order to reprocuce the virtual environment used in this experimentation, conda need to be installed (it can be installed from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)) and an environment need to be created as follows:

```
conda env create -f environment.yml
```
With ```environment.yml``` the file in the root of the repository. By doing this, all packages are also imported to the created environment.

And that's it!

## The Notebook

The notebook is divided into sections:
- <b>Introduction</b><br/>
Here the model is defined, its dataset is loaded, and the model is trained or it's weights can be loaded if you want to use the same model used during the experimentation.
- <b>Model performances</b><br/>
Here model's performances are evaluated with metrics like accuracy, precition, recall, ecc...
- <b>ZOO Attack</b><br/>
Here the zoo attack is performed in it's targeted and untargeted version
- <b>Bayesian Optimization for ZOO attack</b><br/>
Here the bayesian optimization for untargeted and targeted ZOO is implemented
- <b>BadNets</b><br/>
here BadNets attack is implemented with the whole grid search process used to find best values for it's parameters.
- <b>Clean-label backdoor poisoning attack</b><br/>
Here the clean-label black-box attack is implemented.
- <b>Other solution tried during the experimentation</b><br/>
Here there are some other things tried during the experimentation which were found to be unsuccessful. 

## License

[MIT](https://choosealicense.com/licenses/mit/)