<!-- PROJECT INTRODUCTION -->
<h3 align="center"> Parametric Linear Optimization Using Neural Networks</h3>

  <p align="center">
    This project covers the necessary implementation for the definition of a feedforward neural network for learning the required mapping, 
    which I have described in my thesis "Parametric Linear Optimization using Neural Networks", 
    including the necessary functions for data processing.
    <br />
    <a href="https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks"><strong>Explore the files »</strong></a>
    <br />
    <a href="https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/issues">Report Bug  »</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Packages</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
       <li>
      <a href="#contact">Contact</a>
      <ul>
    </li>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
This project focuses on using a Feedforward Neural Network (FNN) to learn a mapping from a parameter to the optimal solution set of a parameterized linear optimization problem.
The tasks of each file are summarized as follows <br />
* [FNN.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/FNN.py) :
  Defines the FNN class and necessary functions in order to be able to process and store the data as well as evaluating the perfomance of the model.
* [FNN_Learning.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/FNN_Learning.py) :
  Here, we train and evaluate the model as well as store the generated training and test data sets along with the model's predictions.
  We also tune the model manually, e.g. by testing different network structures for the same data set.
* [Generate_Dataset_Mozart.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Generate_Dataset_Mozart.py) :
  Here, we simply generate the data set of the parameterized version of the Mozart problem that we want to consider during training,
  as described in my thesis in Section 3.1.
* [Generate_Dataset_Transport.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Generate_Dataset_Transport.py) :
  Here we proceed in an analogous way, only for the transport problem described in Section 3.2 of my thesis.
* [Plot_Feasible_Region_Mozart.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Plot_Feasible_Region_Mozart.py) :
  This file is used to visualize the feasible region of the Mozart problem.
* [Plot_R2_Scores_And_Losses.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Plot_R2_Scores_And_Losses.py) :
  This file creates a plot showing the differences in loss or R2 over time (use the corresponding file created via the train_and_evaluate_model() in FNN.py) between the test and training sets.
* [Plot_ReLU.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Plot_ReLU.py) :
  This visualizes the ReLU function and its derivative.
* [Plot_Targets_And_Predictions.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Plot_Targets_And_Predictions.py) :
  This file generates plots that illustrate the relationship between the parameters and the optimal solution.
  One plot uses the targets and the other uses the predictions to compare them.
* [Prepare_Data_for_Plots.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Prepare_Data_for_Plots.py) :
  This is where the data stored during training is prepared for the plot showing the targets and predictions by assigning the individual outputs to their inputs.


### Built With

* [PyCharm](https://www.jetbrains.com/pycharm/)
* [Python 3.7](https://www.python.org/downloads/release/python-370/)
* [PyTorch](https://pytorch.org/)


<!-- GETTING STARTED -->
## Getting Started

Here are instructions on which packages to download and in what order to use the files.

### Packages

The following packages are needed: pandas, torch, numpy, tqdm, scikit-learn, matplotlib, cvxpy, plotly.


### Usage

1. The first step is to generate the data that will be used to train the network. 
This project contains two example problems 
[Generate_Dataset_Mozart.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Generate_Dataset_Mozart.py) 
and [Generate_Dataset_Transport.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Generate_Dataset_Transport.py) 
that you can use to generate the corresponding data. 
2. In [FNN_Learning.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/FNN_Learning.py) 
you can now set the size of your data set, which equations you have parameterized, and which network structure (model_no) you are using, 
before you run the program and train the network.
3. Then you need to prepare the data for 4. in [Prepare_Data_for_Plots.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Prepare_Data_for_Plots.py).
4. Now you can visualize the losses and R2 Scores with 
[Plot_R2_Scores_And_Losses.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Plot_R2_Scores_And_Losses.py)
and compare the targets with the predictions with 
[Plot_Targets_And_Predictions.py](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks/blob/main/Plot_Targets_And_Predictions.py).

<!-- CONTACT -->
## Contact

Karina Kniel - karina.kniel@stud.uni-heidelberg.de

Project Link: [https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks](https://github.com/KarinaKniel/ParametricLinearOptimizationUsingNeuralNetworks)
