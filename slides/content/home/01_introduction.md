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
### How does it work?

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
## Neural Networks
* It appears that one reason why the human brain is <em>so powerful</em> is the
sheer complexity of connections between neurons;
* The brain exhibits <em>huge degree of parallelism</em>;
<center><img src="neurons.png" width="600px"/></center>

---

## Neural Networks
* Model each part of the neuron and interactions;
* <em>Interact multiplicatively</em> (e.g. $w_0x_0$) with the dendrites of the other neuron based
on the synaptic strength at that synapse (e.g. $w_0$ ); 
* Learn <em>synapses strengths</em>;

---

## Neural Networks
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

## Try to train it manually:

<iframe src="manual_NN1.html" height="500px" width="800px">
</iframe>

###### Widget taken from Jay Alammar <a href="https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/" target="_blank">blog post</a>





