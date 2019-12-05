## Neural Networks

* Neurons as structural constituents of the brain [Ramón y Cajál, 1911]; 
* Five to six orders of magnitude slower than silicon logic gates; 
* In a silicon chip happen in the nanosecond on chip) vs millisecond range (neural events); 
* A truly staggering number of neurons (nerve cells) with massive interconnections between them;

---

## Neural Networks 

* Receive input from other units and decides whether or not to fire;
* Approximately 10 billion neurons in the human cortex, and 60 trillion synapses or connections [Shepherd and Koch, 1990];
* Energy efficiency of the brain is approximately $10^{−16}$ joules per operation per second against ~ $10^{−8}$ in a computer;

---

{{<slide background-image="neuron2.png">}}
## Neurons
---

## Neurons

* input signals from its <em>dendrites</em>;
* output signals along its (single) <em>axon</em>;

<img src="neuron1.png"/>


---
## Neurons
### How do they work?

<ul>
<div align="left">
{{% fragment %}} <li>Control the influence from one neuron on another:</li> {{% /fragment %}}
</div>

<ul>
<div align="left">
{{% fragment %}} <li><em>Excitatory</em> when weight is positive; or</li> {{% /fragment %}}
</div>

<div align="left">
{{% fragment %}} <li><em>Inhibitory</em> when weight is negative;</li> {{% /fragment %}}
</div>
</ul>

<div align="left">
{{% fragment %}} <li>Nucleus is responsible for summing the incoming signals;</li> {{% /fragment %}}
</div>

<div align="left">
{{% fragment %}} <li><strong>If the sum is above some threshold, then <em>fire!</em></strong></li> {{% /fragment %}}
</div>
</ul>

---
## Neurons
### Artificial Neuron

<center><img src="artificial_neuron.jpeg" width="800px"/></center>

---
{{<slide background-image="neurons.png">}}
## Neural Networks

---
## Neural Networks
* It appears that one reason why the human brain is <em>so powerful</em> is the
sheer complexity of connections between neurons;
* The brain exhibits <em>huge degree of parallelism</em>;

---

## Artificial Neural Networks
* Model each part of the neuron and interactions;
* <em>Interact multiplicatively</em> (e.g. $w_0x_0$) with the dendrites of the other neuron based
on the synaptic strength at that synapse (e.g. $w_0$ ); 
* Learn <em>synapses strengths</em>;

---

## Artificial Neural Networks
### Function Approximation Machines
* Datasets as composite functions: $y=f^{*}(x)$
  * Maps $x$ input to a category (or a value) $y$;
* Learn synapses weights and aproximate $y$ with $\hat{y}$:
  * $\hat{y} = f(x;w)$
  * Learn the $w$ parameters; 

---

## Artificial Neural Networks
* Can be seen as a directed graph with units (or neurons) situated at the vertices;
  * Some are <em>input units</em>
* Receive signal from the outside world;
* The remaining are named <em>computation units</em>;
* Each unit <em>produces an output</em>
  * Transmitted to other units along the arcs of the directed graph;

---
## Artificial Neural Networks
* <em>Input</em>, <em>Output</em>, and <em>Hidden</em> layers;
* Hidden as in "not defined by the output";
<center><img src="nn1.png" height="200px" style="margin-top:50px;"/></center>

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* Imagine that you want to forecast the price of houses at your neighborhood;
  * After some research you found that 3 people sold houses for the following values:

<br />

Area (sq ft) (x)|	Price (y)
----------------|----------
2,104	          |  $399,900$
1,600	          |  $329,900$
2,400	          |  $369,000$

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 

{{% fragment %}} If you want to sell a 2K sq ft house, how much should ask for it? {{% /fragment %}}
<br /><br />
{{% fragment %}} How about finding the <em>average price per square feet</em>?{{% /fragment %}}
<br /><br />
{{% fragment %}} <em>$\$180$ per sq ft.</em> {{% /fragment %}}


---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* Our very first neural network looks like this:
{{% fragment %}}<center><img src="nn2.png" width="600px"/></center> {{% /fragment %}}

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* Multiplying $2,000$ sq ft by $180$ gives us $\$360,000$. 
* Calculating the prediction is simple multiplication. 
* <strong><em>We needed to think about the weight we’ll be multiplying by.</em></strong>
* That is what training means!

<br />

Area (sq ft) (x)|	Price (y)    | Estimated Price($\hat{y}$)
----------------|--------------|---------------------------
2,104	          |  $\$399,900$ |          $\$378,720$
1,600	          |  $\$329,900$ |          $\$288,000$
2,400	          |  $\$369,000$ |          $\$432,000$

---
## Artificial Neural Networks
###### Motivation Example (taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>) 
* How bad is our model?
	* Calculate the <em>Error</em>;
	* A better model is one that has less error; 

{{% fragment %}} <em>Mean Square Error</em>{{% /fragment %}}{{% fragment %}}: $2,058$ {{% /fragment %}}

<br />

Area (sq ft) (x)|	Price (y)    | Estimated Price($\hat{y}$) | $y-\hat{y}$ | $(y-\hat{y})^2$
----------------|--------------|----------------------------|-------------|---------------
2,104	          |  $\$399,900$ |          $\$378,720$       | $\$21$      |  $449$
1,600	          |  $\$329,900$ |          $\$288,000$       | $\$42$      |  $1756$
2,400	          |  $\$369,000$ |          $\$432,000$       | $\$-63$     |  $3969$

---
## Artificial Neural Networks
* Fitting the line to our data:

<center><img src="manual_training1.gif" width="450px"/></center>

Follows the equation: $\hat{y} = W * x$

---
## Artificial Neural Networks

How about addind the <em>Intercept</em>?

{{% fragment %}} $\hat{y}=Wx + b$ {{% /fragment %}}

---
## Artificial Neural Networks
### The Bias

<center><img src="nn3.png" width="500px"/></center>

---
## Artificial Neural Networks
### Try to train it manually:

<iframe src="manual_NN1.html" height="500px" width="800px">
</iframe>

---
## Artificial Neural Networks
### How to discover the correct weights?
* Gradient Descent:
  * Finding the <em>minimum of a function</em>;
    * Look for the best weights values, <em>minimizing the error</em>;
  * Takes steps proportional to the negative of the gradient of the function at the current point.
  * Gradient is a vector that is tangent of a function and points in the direction of greatest increase of this function. 

---
## Artificial Neural Networks
### Gradient Descent
* In mathematics, gradient is defined as partial derivative for every input variable of function;
* Negative gradient is a vector pointing at the greatest decrease of a function;
* Minimize a function by iteratively moving a little bit in the direction of negative gradient;

---
## Artificial Neural Networks
### Gradient Descent
* With a single weight: 

<center><img src="gd1.jpeg" width="500px"/></center>


---
## Artificial Neural Networks
### Gradient Descent

<iframe src="manual_NN2.html" height="500px" width="800px">
</iframe>


---
## Artificial Neural Networks
### Perceptron
* In 1958, Frank Rosenblatt proposed an algorithm for training the perceptron.
* Simplest form of Neural Network;
* One unique neuron;
* Adjustable Synaptic weights

---
## Artificial Neural Networks
### Perceptron
* Classification of observations into two classes:
<center><img src="perceptron1.png" height="350px"/></center>

###### Images Taken from <a href="https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975" target="_blank">Towards Data Science</a> 

---
## Artificial Neural Networks
### Perceptron
* Classification of observations into two classes:
<center><img src="perceptron2.png" height="350px"/></center>

###### Images Taken from <a href="https://towardsdatascience.com/perceptron-learning-algorithm-d5db0deab975" target="_blank">Towards Data Science</a> 

---
## Artificial Neural Networks
### Perceptron
* E.g, the OR function:

<center><img src="or1.png" width="550px"/></center>

#### Find the $w_i$ values that could solve the or problem. 

---
## Artificial Neural Networks
### Perceptron
* E.g, the OR function:

<br />
<center><img src="or2.png" width="550px"/></center>

---
## Artificial Neural Networks
### Perceptron
* One possible solution $w_0=-1$, $w_1=1.1$, $w_2=1.1$:

<center><img src="or3.png" width="450px"/></center>

---
## Artificial Neural Networks
### Neurons
* Activation Function
  * Describes <em>whether or not the neuron fires</em>, i.e., if it forwards its value for the next neuron layer;
* <em>Multiply the input</em> by its <em>weights</em>, <em>add the bias</em> and <em>applies activation</em>;
* Sigmoid, Hyperbolic Tangent, Rectified Linear Unit;
  * Historically they translated the output of the neuron into either 1 (On/active) or 0 (Off)

---
## Artificial Neural Networks
### The Bias
<center><img src="neuron3.png" width="650px"/></center>

---

## Artificial Neural Networks
### The Bias
<center><img src="neuron4.png" width="650px"/></center>

---
## Artificial Neural Networks
### The Bias
<center><img src="bias1.png" width="600px"/></center>

---
## Artificial Neural Networks
### The Bias
<center><img src="bias2.png" width="600px"/></center>





