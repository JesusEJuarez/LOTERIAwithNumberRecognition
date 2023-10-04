# Mexican Lottery Card Recognition

## Introduction

Mexican Lottery Card Recognition is a project that aims to recognize and interpret Mexican lottery cards, a popular game of chance similar to bingo. The project uses various image processing techniques and an artificial neural network to achieve this goal.

### What is Mexican Lottery?

Mexican Lottery is a game of chance similar to bingo, played with a deck of cards containing numbers and representative drawings. The deck consists of 54 different images, including objects and characters such as the mermaid, the dandy, the barrel, the star, and the cactus. The objective of the game is to fill a card (also known as a "board") chosen by the player, completing a certain pattern (such as four images in a row, diagonally, or in the corners). The first person to complete their card shouts "¡lotería!" to end the round.

During the game, there is a person called "el gritón" who sings the cards, often accompanied by a phrase or saying related to the image. For example, the shrimp card might be followed by the saying "camarón que se duerme se lo lleva la corriente," while the little devil card could be followed by "pórtate bien cuatito, si no te lleva el coloradito."

In this project, to preprocess Mexican lottery cards, filters were used along with a neural network for recognition.

## Filters

### Erosion Filter
This filter involves removing all pixels from an object whose neighborhood contains at least one pixel belonging to the background. Its effect is to reduce the size of objects or eliminate very small ones. It is represented by the expression I $\theta$ H, where I represents the image and H is a given neighborhood.

### Dilation Filter
All background pixels in whose neighborhood there is at least one pixel belonging to the object are turned into an object (Figure 2). Its effect is to enlarge objects or close very small holes. It is represented by the expression I \oplusH, where I represents the image and H is a given neighborhood.

### Median Filter
The median filter is an example of a linear filter. It basically replaces each pixel in the output image with the mean (average) value of the neighborhood. This has the effect of smoothing the image (reducing the intensity variations between adjacent pixels), removing image noise, and brightening the image.

### Average Filter (CV2)
The average is calculated by convolving the image with a normalized box filter. Thus, this filter takes the average of all pixels under the kernel area and replaces the central element with this average. A normalized 3x3 box filter would look like this:

### Artificial Neural Network

To recognize the numbers on the Mexican lottery cards, a neural network was trained.

#### Artificial Neuron
An artificial neuron can be thought of as the sum of the inputs of a polynomial, which in our case is related to the communication of brain neurons. It is the sum of the inputs multiplied by their associated weights, determining the "nerve impulse" received by the neuron. This value is processed inside the cell through an activation function that returns a value sent as the neuron's output.

It is also important to mention that neurons can be interconnected with each other. An artificial neural network can be formed by connections in series or in parallel, and these connections are grouped into different layers called layers.

#### Backpropagation
Neurons, as explained above, work only in a forward sequence propagation, but in this configuration, forward and backward information propagation will be performed. The following steps of the propagation algorithm are:

##### Forward Propagation
1. $a^o = P$ (where $a^o$ is the initial pattern $P$).
2. $a^{m+1} = f^{m+1}(W^{m+1}a^m + b^{m+1})$ for $m=0,1,2,...,M-1$ (where $a^{m+1}$ is the output of the current layer, $W^{m+1}$ is the weight matrix, $b^{m+1}$ is the bias vector, and $f^{m+1}$ is the activation function).
3. $a^M = f^M(W^Ma^{M-1}+b^M)$ (where $a^M$ is the final result).

##### Backward Propagation
1. Compute $\delta^M = -2 \cdot \mathcal{E}$ (where $\delta^M$ is the error term, and $\mathcal{E}$ is the error).
2. Compute $\delta^{m-1} = (W^{m})^T\delta^{m+1} \odot f'(W^{m}a^m + b^{m})$ (where $\odot$ represents element-wise multiplication and $f'$ is the derivative of the activation function).
3. Update $W^m = W^m - \alpha\delta^{m+1}a^{m^T}$ and $b^m = b^m - \alpha\delta^{m+1}$ (where $\alpha$ is the learning rate).

## Development

### Reading Cards

The project uses a camera to capture images of Mexican lottery cards. These images are then processed using various filters to prepare them for recognition.

### Training Code

To recognize the numbers on the cards, an artificial neural network is trained. The training code takes a dataset of labeled images and adjusts the network's weights and biases to minimize the error between its predictions and the actual labels.

### Line Code

The core functionality of the game is implemented in the `loteria.py` file. This file contains the main loop of the game, which allows the player to take photos of cards and checks for a win condition.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Make sure you have the required dependencies installed.
3. Connect a webCam, we recomend using a controled light an a sopport for the cards.
4. Run the `loteria.py` file to start the game.
5. Follow the on-screen instructions to play the game.
## Results 
You can check a test video in the next link:
https://www.youtube.com/watch?v=4C27O5_hgO0
## Autors 
JesusEJuarez, Catherin4 and NahoryRamirez
