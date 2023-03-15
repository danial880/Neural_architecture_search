# Neural_architecture_search
Neural architecture search (NAS) refers to the process of automatically discovering the optimal neural network architecture for a given task. In other words, it is a technique for automating the design of neural networks, which traditionally requires significant expertise and trial-and-error experimentation.

NAS algorithms typically involve searching through a large space of potential neural network architectures using various optimization techniques such as reinforcement learning, evolutionary algorithms, and gradient-based methods. These algorithms can explore a vast search space, including different types of layers, activation functions, and connections between layers, to find the optimal architecture for a given task.

NAS has the potential to significantly accelerate the development of new neural network models for various applications, such as image classification, object detection, speech recognition, and natural language processing. With NAS, researchers can efficiently explore the space of possible neural network architectures and find the best-performing architecture for a given task.

## Configuration
The configuration file is created as the [config.yaml](./config.yaml) is contains the following fields.

*   dataset_to_run: 'name of the dataset'	
*   datasets: ' contains the paremeters of the datasets

Here in the configuration we need to specify dataset name to search the model architecture. The list of dataset is as folliwing `</DTD, ST10, CIFAR100, Food101 and Flower102/>` objects.

## How to run
To run the neural architecture search code use the [main.py](./main.py). This file needs the following arguments.
*  data_dir <Path of the directory containing the dataset>
*  batch_size < specify batch size>
*  epochs < specify number of epoch>
*  save <Path to save experiment>

We can use the follwing command to run the code.

 `python main.py--batch_size 32 --epochs 100 --save cifar100_experiment`
 
