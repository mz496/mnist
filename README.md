# MNIST handwritten digit reimplementation

- [x] Set up Hugging Face data ingest + W&B logging
- [x] Set up train loop
- [x] Set up validation loop 
- [x] Set up test loop
- [ ] Visualize loss landscape and the zigzagging trick (LeCun 1998)

## Things I learned

(and/or did not think about the last time I implemented something like this in ~2017)

- Thinking a lot more about types and truly understanding the matrix math. I spend much more time thinking about types. Pytorch type stubs are pretty bad
- Lazy loading, CPU/GPU data transfer somewhat nonintuitive; dataset is probably still too small to see actual differences here
- W&B is nice for plotting/logging
- Role of AI - I basically went back to 2023 in some ways, not allowing the AI to write code for me unless it was generic stuff I don't really care about (like matplotlib libraries, which I sped through ie letting a second agent rip while I was asking a main agent questions). But better since it knows my context, can read the other reference repo, etc.