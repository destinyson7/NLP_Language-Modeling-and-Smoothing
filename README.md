
Language Modeling and Smoothing
==================
Given a training corpus corpus.txt, use it to create an n-gram language model, where n can be provided as a parameter. Perform smoothing on the language model using:
+ Witten Bell Smoothing, and
+ Kneser Ney Smoothing
  
### How to Run:
```python
python3 language_model.py <value of n> <smoothing type> <path to input corpus>
```
where n can be between 1 and 3, and smoothing type can be k for Kneser Ney or w for Witten Bell.

#### Q.  Compare the models of smoothing and explain in which cases the outputs of the two smoothing mechanisms differ and why

The Kneser Ney smoothing is an extension of absolute discounting with a clever way of constructing the lower-order (backoff) model. The idea is that the lower-order model is significant only when count is smaller or zero in the higher-order model, and so should be optimized for that purpose. Ex: “San Francisco” is common, but “Francisco”occurs only after “San”. So “Francisco” will get a high unigram probability, and so absolute discounting will give a high probability to “Francisco” appearing after novel bigram histories. It is better to give “Francisco” a low unigram probability, because the only time it occurs is after “San”, in which case the bigram model fits well.

The motivation of the Witten Bell smoothing is to interpret λ as the probability of using the higher-order model. We should use higher-order model if the n-gram was seen in the training data, and back off to lower-order model otherwise. So 1−λ should be the probability that a word not seen after the (n-1) gram in the training data occurs after that history in test data. We estimate this by the number of unique words that follow the history i.e. (n-1) gram in the training data.

After rigorous testing on this simple language model, it has been observed that the Kneser Ney technique consistently outperforms the Witten Bell technique. Witten Bell Smoothing is more conservative when subtracting probability mass and gives good probability estimates. Kneser Ney discounting augments absolute discounting with a more sophisticated way to handle the backoff distribution using linear interpolation and gives even better estimates than the Witten Bell Smoothing in most of the cases.
