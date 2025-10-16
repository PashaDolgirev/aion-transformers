Transformer from Scratch

I am a theoretical physicist exploring modern AI architectures through first-hand prototyping.
This repository follows the superb lecture by Andrej Karpathy (https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5065s), where I build a Transformer model from scratch — line by line — to internalize the self-attention mechanism in full detail. I am exploring if this mechanism, as well as other NN architectures (see my locality repo for an example), can be leveraged for my physics projects on quantum material design and processing quantum simulators data. Shoot me an email (p_dolgirev@g.harvard.edu) if you have great insighs and want to collab.


📚 Foundational Readings

While training neural networks, I found the following works particularly illuminating:
1. Deep Residual Learning for Image Recognition (He et al., 2015, https://arxiv.org/abs/1512.03385). Key lesson: naively, one expects that a deeper NN with more model parameters would result in overfitting. This is not what is observed in practice. In practice, deeper NNs are more difficult to train, and the proposed residual architecture efficiently solves the training issue.
2. Attention Is All You Need — Vaswani et al., 2017, https://arxiv.org/abs/1706.03762. The original breakthrough paper introducing the Transformer architecture, based on the (self) attention mechanism, note the positional encoding.
3. 3Blue1Brown: Transformer Series — Chapters 5–7, https://www.youtube.com/watch?v=wjZofJX0v4M. A beautifully intuitive explanation of the self-attention mechanism, really complements the Attention Is All You Need paper.
4. Murphy, Probabilistic Machine Learning: An Introduction — Chapters 13–15: concise and complete overview of deep learning fundamentals, love the rigor level of the book.
