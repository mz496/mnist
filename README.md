# MNIST handwritten digit reimplementation

- [x] Set up Hugging Face data ingest + W&B logging
- [x] Set up train loop
- [x] Set up validation loop 
- [x] Set up test loop
- [ ] Visualize loss landscape and the zigzagging trick (LeCun 1998)

## Things I learned

(and/or did not think about the last time I implemented something like this in ~2017)

- Pytorch type stubs are pretty bad
- Lazy loading, CPU/GPU data transfer somewhat nonintuitive; dataset is probably still too small to see actual differences here
- W&B is nice for plotting/logging
