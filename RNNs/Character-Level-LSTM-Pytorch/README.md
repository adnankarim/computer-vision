# Character-Level LSTM in PyTorch

In this notebook, I'll construct a character-level LSTM with PyTorch. The network will train character by character on some text, then generate new text character by character. As an example, I will train on Anna Karenina. **This model will be able to generate new text based on the text from the book!**

This network is based off of Andrej Karpathy's [post on RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and [implementation in Torch](https://github.com/karpathy/char-rnn). Below is the general architecture of the character-wise RNN.

<img src="https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/855fb9a8f125bec59fb18f574a771107ff9ddba1/recurrent-neural-networks/char-rnn/assets/charseq.jpeg" width="500">