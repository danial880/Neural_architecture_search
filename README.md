# Neural_architecture_search
Neural architecture search (NAS) refers to the process of automatically discovering the optimal neural network architecture for a given task. In other words, it is a technique for automating the design of neural networks, which traditionally requires significant expertise and trial-and-error experimentation.

NAS algorithms typically involve searching through a large space of potential neural network architectures using various optimization techniques such as reinforcement learning, evolutionary algorithms, and gradient-based methods. These algorithms can explore a vast search space, including different types of layers, activation functions, and connections between layers, to find the optimal architecture for a given task.

NAS has the potential to significantly accelerate the development of new neural network models for various applications, such as image classification, object detection, speech recognition, and natural language processing. With NAS, researchers can efficiently explore the space of possible neural network architectures and find the best-performing architecture for a given task.

## Configuration
The configuration file is created as the [config.yaml](./config.yaml) is contains the following fields.

*   dataset_to_run: name of the dataset	
*   datasets: contains the paremeters of the datasets
*   hyperparameters: training parameters
*   search_parameters: important arguments for network search
*   random_network_gen: parameters for generating random networks
*   custom_network: parameters for training a custom network

Here in the configuration we need to specify dataset name to search the model architecture. Supported datasets `DTD, ST10, CIFAR10, CIFAR100, Food101 and Flowers102` objects.

## How to run
##### Neural architecture search
`python main.py` 
##### Random network generator
`python rnd_ntw_gen.py` 
##### Train custom network
`python train_custom_net.py`
##### Parameter Count and Latency
`python params.py`
Calculate custom network parameters and latency
