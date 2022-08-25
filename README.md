![](https://github.com/rohan1198/NeRF/edit/main/output.png)
Output

<p align="center"><b><ins> NeRF Notes </ins></b></p>

- In the paper [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/pdf/2003.08934.pdf), the authors have described a method for novel view synthesis from a sparse set of input images.
- The algorithm proposed by the authors represents a scene using a <ins>non-convolutional fully-connected</ins> deep network.

<br><br>

- The input is a <ins>single continuous 5D coordinate</ins>, comprising of the <ins>spatial location</ins> (x, y, z) and the <ins>viewing direction</ins> (θ, φ).
- The output is the <ins>volume density</ins> and view-dependent <ins>emitted radiance</ins> at a particular spatial location.
- The 5D coordinates are queried along the camera rays, and the output (colours and densities) is projected to an image using classic volume rendering techniques.
<br><br>
- Since <ins>volume rendering is naturally differentiable</ins>, the only input required is a set of images and their corresponding camera poses.

---

<br><br>
<b><ins> Introduction </b></ins>

- A static scene is represented as a continuous 5D function that outputs the <ins>radiance emitted</ins> in each direction (θ, φ) at each point (x, y, z) in space and a <ins>density at each point</ins> which acts as a differential opacity controlling how much radiance is accumulated by a ray passing through (x, y, z).
- The method proposed in this paper optimizes a multilayer perceptron (MLP) network (without the use of any convolutional neural networks) to represent the 5D function.
- This is done by regressing from a single 5D coordinate (x, y, z, θ, φ) to a <ins>single volume density</ins> and <ins>view dependent RGB colour</ins>.
- To render this <i>neural radiance field </i> from a particular point, <br>
	i. Camera rays are marched throught the scene to generate a sampled set of 3D points. <br>
	ii. These points and their corresponding 2D viewing directions are used as inputs to the neual network to produce an output set of colours and densities. <br>
	iii. Classical volume rendering techniques are used to accumulate the colours and densities into a 2D image.

<br>

- Since this method is naturally differentiable, gradient descent an be used to optimize the model by <ins>minimizing the error between each observed image and the corresponding rendered views</ins>.
- Minimizing this error across <ins>multiple views</ins> encourages the network to predict a coherent model of the scene by assigning high volume densities and accurate colours to the locations that contain the true underlying scene content.

<br>

- The authors found that the basic implementation of optimizing a neural radiance field for a complex scene does not converge to a sufficiently high resolution representation and is inefficient in the number of samples per camera ray.
- To address these issues, the authors propose transforming the input 5D coordinates with a <ins>positional encoding</ins> to enable the MLP to represent higher frequency functions.
- Additionally, a <ins>heirarchical sampling</ins> is proposed to reduce the number of queries required to adequately sample the high frequency scene representation.

<br>

- The proposed approach can represent complex real-world geometry and appearance, and it is well suited for gradient-based optimization using projected images.
- More importantly, the proposed aproach overcomes the storage costs of discretized voxel grids when modelling complex scenes at a high resolution.

---

<b><ins>Neural Radiance Field Scene Representation</ins></b>

- A continuous scene is represented as a 5D vector-valued function whose input is a <ins>3D location x = (x, y, z)</ins> and <ins>2D viewing direction (θ, φ)</ins>, and whose <ins>output is an emitted color c = (r, g, b) and volume density $\sigma$</ins>.

- The direction is expressed as a 3D Cartesian unit vector <i>d</i>.

- Then, the continuous 5D scene representation is approximated with an MLP by:

$$ F_{\theta}: (x, d) \rightarrow (c, \sigma)$$

- The weights $\theta$ can then be optimized to map each input 5D coordinate to its corresponding volume density and directional emitted color.

<br><br>

- To make the representation multiview consistent, the authors restricted the network to predict the <ins>volume density $\sigma$</ins> as a function of <ins>only the location $x$</ins>, while allowing the <ins>RGB color $c$</ins> to be predicted as a function of <ins>both location and viewing direction</ins>.

<br>

- To accomplish this, the MLP $F_{\theta}$ first processes the input 3D coordinate with <ins>8 fully connected layers</ins> (using ReLU activation functions and 256 channels per layer), and outputs $\sigma$ and a 256-dimensional feature vector.
- This feature vector is then concatenated with the camera ray's viewing direction and passed to one additional fully-connected layer (using ReLU activation and 128 channels), which outputs the view-dependent RGB colour.

---

<b><ins>Volume Rendering with Radiance Fields</b><ins>

- The proposed 5D neural radiance field represents a scene as the volume density and directional emitted radiance at any point in space.
- This volume density $\sigma(x)$ can be interpreted as the <ins>differential probability of a ray terminating at an infinitesimal particle location $x$</ins>.
- The expected color $C(r)$ of a camera ray $r(t) = o + td$ with near and far bounds $t_{n}$ and $t_{f}$ is given by:

$$ C(r) = \int_{t_{n}}^{t_{f}} T(t)\sigma(r(t))c(r(t), d)dt $$

where,

$$ T(t) = {e}^{-\int_{t_{n}}^{t} \sigma(r(s))ds} $$

- The function $T(t)$ denotes the <ins>accumulated transmittance</ins> along the ray from $t_{n}$ to $t$.
- In other words, the function denotes the <ins>probability that the ray travels from $t_{n}$ to $t$ without hitting any other particle</ins>.

<br><br>

- Rendering a view from the continuous neural radiance field requires estimating the integral $C(r)$ for a camera ray traced through each pixel of the desired virtual camera.
- This continuous integral is numerically estimated using quadrature. (Deterministic quadrature, which is typically used for rendering discretized voxel grids, would effectively limit the representation's resolution as the MLP would only be queried at fixed set of locations.)
- A stratified sampling approach is used where $[t_{f}, t_{n}]$ are partitioned into N evenly-spaced bins, and one sample is drawn at random from within each bin.

---

<b><ins>Optimizing a Neural Radiance Field</b></ins>

- In a nutshell, positional encoding of the input coordinates assists the MLP in representing high-frequency functions, and a heirarchical sampling procedure allows efficient sampleing and high-frequency representation.

<br><br>

<b> Positional Encoding </b>

- [Deep neural networks are inherently biased towards learning lower frequency functions.](https://arxiv.org/pdf/1806.08734.pdf)
- Additionally, the mapping of inputs to a higher dimensional space using high frequency functions before passing them to the network enables beter fitting of the data that contains high frequency variation.

<br>

- Using these findings, it is shown that reformulating $F_{\theta}$ as a composition of two functions $F_{\theta} = F'_{\theta} \bullet \gamma$, one learned and one not, significantly improves performance.
- Here, $\gamma$ is a mapping from $\mathbb{R}$ into a higer dimensional space $\mathbb{R}^{2L}$, and $F'_{\theta}$ is simply a regular MLP.
- The encoding function used is:

$$ \gamma(p) = (sin(2^{0}\pi p), cos(2^{0}\pi p), ..., sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))$$

<br>

- <i>Positional encoding</i> was originally proposed for Transformer architectures to provide the discrete positions of tokens in a sequence as input to an architecture that does not contain any notion of order.
- In contrast, these functions are used to map continuous input coordinates into a higher dimensional space to enable the MLP to more easily approximate a higher frequency function.

<br><br>

<b>Heirarchical Volume Sampling</b>

- The rendering strategy proposed by the authors was inefficient as free space and occluded regions that did not contribute to the rendered image were still sampled repeatedly.
- To counter thism aa heirarchical representation was proposed to increase the rendering effiniency by allocating samples proportionally to their expected effect on the final rendering.

<br>

- Two networks, "<i>fine</i>" and "<i>coarse</i>" are optimized simultaneously.
- First a set of $N_{c}$ locations were sampled using stratified sampling, then a "coarse" network was evaluated at these locations.
- Given the output of this "coarse" network, a more informed sampling of points are produced along each ray where the samples are biased towards the relevant parts of the volume.
- To do this, te alpha composited colour is first rewritten from the coarse network as a weighted sum os all the sampled colours $c_{i}$ along the ray:

$$ \hat{C}_{c}(r) = \sum_{i=1}^{N_{c}}w_{i}c_{i} $$
	
where,

$$ w_{i} = T_{i}(1 - \exp(-\sigma_{i}\delta_{i})) $$

<br>

- Normalizing these weights as produces a piecewise-constant [Probability Density Function](https://en.wikipedia.org/wiki/Probability_density_function) (PDF) along the ray:
	
$$ \hat{w}_{i} = \frac{w_{i}}{\sum_{j=1}^{N_{c}} w_{j}}$$  
	
- A second set of $N_{f}$ locations are sampled from this distribution using [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling).
- The "fine" network is evaluated at the union of the first and second set of samples and the final rendered colour of the ray is computed using all $N_{c} + N_{f}$ samples.
- This procedure allocates more samples to regions expected to contain visible content.
