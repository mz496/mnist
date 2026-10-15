# MNIST handwritten digit reimplementation

- [x] Set up Hugging Face data ingest + W&B logging
- [x] Set up train loop
- [x] Set up validation loop 
- [x] Set up test loop
- [x] Visualize loss landscape

## Things I learned

(and/or did not think about the last time I implemented something like this in ~2017)

- Thinking a lot more about types and truly understanding the matrix math. I spend much more time thinking about types. Pytorch type stubs are pretty bad
- Lazy loading, CPU/GPU data transfer somewhat nonintuitive; dataset is probably still too small to see actual differences here
- W&B is nice for plotting/logging
- Role of AI - I basically went back to 2023 in some ways, not allowing the AI to write code for me unless it was generic stuff I don't really care about (like matplotlib libraries, which I sped through ie letting a second agent rip while I was asking a main agent questions). But better since it knows my context, can read the other reference repo, etc.


## Log

I started out with a goal:
- Build understanding and intuition of a problem
- Learn how to use modern tooling
- Build the skill of asking questions and answering them

Using the handwritten digit classification task on MNIST as a testbed.


### Oct 7

Set up Hugging Face, Weights & Biases, ipynb workflow.

### Oct 8

Laid out the LeNet5 model architecture from the paper.

Spent a while refreshing my memory on each operation and what it does to the activation dimensions and parameter count (convolution, flatten, fully connected layers).

### Oct 9

Added typechecking, data loaders, config infra, randomness seeds.

Spent a while on types to make sure I have everything straight on my data preparation:
- The Hugging Face `dataset` library's `Dataset` type vs. the PyTorch library's `Dataset` type.
  - I ended up aliasing them `ArrowDataset` and `TorchDataset` respectively for clarity and somewhat giving up on typing, just doing a `cast()` instead. They implement the same `__len__` and `__getitem__` so this is fine anyway. Not worth being super precise.
- I did want to document the type of the HF dataset entries, hence the `InputDatasetItem`.
- I lazyload the dataset to build a good habit even though MNIST fits in memory.
    - Given this - where does the input transform live, since the images are 0-255 ints, and we want them to be floats? I end up using PyTorch's `collate`. `set_transform` belongs to HF but I would rather work more in PyTorch
- While digging through and justifying each line, Claude suggests I normalize the inputs to the mean, and points me to [Efficient Backprop (LeCun et al, 1998)](https://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf) section 4.3: Normalizing the Inputs). I'm intentionally leaving it out just to see what would happen later. Maybe later I'll also experiment with different hyperparameters like the filter sizes, nonlinearity functions, etc.
- However, I also learned about the "modern" movement toward fiddling less with the exact hyperparameters and trusting the process more, which is simultaneously cool and somewhat maddening that it can be so hands-off. In the original paper, the very specific pattern to the S2-C3 connections ([Gradient-Based Learning Applied to Document Recognition (LeCun et al, 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) section II.B: LeNet5) is rationalized as keeping parameter count down and forcing different features to be learned, which I completely discard as a simplification for now.

### Oct 12

Started doing training. Got my first loss graph logged in W&B; this is a single epoch.

That early plateau is interesting to me, and maybe points to the inefficiency of using a nonoptimal start from not normalizing the input data. Although this is just one seed.

![](img/2026-10-12-train-loss-graph.png)

Also did multiple epochs.

### Oct 13

Started using my validation and test sets and producing plots for validation loss and validation accuracy.

Ran into a few bugs calculating accuracy because I insisted on understanding it myself. Here is where the dimension-level reasoning/typing (`[B, 1, 28, 28]`) was very helpful in debugging, since this was mostly me taking the length of incorrect things. Claude was pretty good at pointing out where I was going wrong.

Out-of-the-box it works decently already, I think? Reached 98.56% validation accuracy, although the differences might be compressed at this scale, where getting to the next 0.5%, 0.25%, etc. gets exponentially harder and harder.

At least the plots aren't showing signs of overfitting, which would look like train loss going down but validation loss staying static if I understand correctly:

![](img/2026-10-13-validation-graphs.png)

At some point I'll set up [type stubs](https://github.com/pytorch/pytorch/wiki/Guide-for-adding-type-annotations-to-PyTorch). Might help with issues similar to when I was trying to pass around a `loss_fn` which was really just cross-entropy loss. I ended up hardcoding a shared type for it so that I could properly annotate what I was actually doing (`(Tensor, Tensor) -> Tensor`).

### Oct 14

Instead of fiddling with hyperparameters yet, I decided to push forward on this other interesting paper I saw in order to grok what is happening a bit better: [Visualizing the Loss Landscape of Neural Nets (Li et al, 2018)](https://ar5iv.labs.arxiv.org/html/1712.09913).

I got 1D plots working first, which roughly matches the graphs from earlier: validation loss and accuracy peak correctly (at a minimum and a maximum respectively) which is a good sign:

![](img/2026-10-14-3ngzl3w2_1d.png)

Then 2D plots with the method of random directions around the best-validation-loss weights. Loss is a bit nicer in that the bands more clearly show the minimum, but probably could have used log-scale for both:

| accuracy | loss |
| --- | --- |
| ![](img/2026-10-14-3ngzl3w2_2d_acc_contour.png) | ![](img/2026-10-14-3ngzl3w2_2d_loss_contour.png) |


I'm also inspired by Chris Olah's attempts at dimensionality reduction to [visualize the dataset itself (2014).](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)

I planned out the changes to plot the training trajectory itself to see if I could replicate or build intuition on that initial plateau from Oct 12.

### Oct 15


