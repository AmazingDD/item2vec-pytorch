# item2vec-pytorch
Notice there are no torch-related code about item2vec, I just want to provide a readable item2vec implementation for researchers

## how to run

The format of input data only need a `pd.DataFrame` with 3 columns(user, item, rating)

Since the paper claims that `Item2Vec` discards time and spatial information of interactions, we can simply fit this model like `word2vec` with some additional preprocessing sampling methods.

To run the code, just enter this code in your terminal
```
python main.py
```

This repository does not provide any predict interface because this model is only to get the **Item Embedding** information after training, then users could do whatever they want flexibly(get rank lists, find KNN items, etc.)

The embedding weight is stored in `model.shared_embedding` with dimension (item_num * embedding_dim)

you can customize the hyperparameter settings by changing the dictionary variable `args` in `main.py`

```
args = {
    'context_window': 2,
    'rho': 1e-5, 
    'batch_size': 256,
    'embedding_dim': 100,
    'epochs': 20,
    'learning_rate': 0.001,
}
```

## References

*Barkan, Oren, and Noam Koenigstein. "Item2vec: neural item embedding for collaborative filtering."* 2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2016.
