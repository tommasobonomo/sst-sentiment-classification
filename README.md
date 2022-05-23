# Stanford Sentiment Treebank sentiment classifier

This repo builds, trains and exposes a system that performs sentiment classification, i.e. identifying how positive a given sentence is.
The system has been trained on the open-source [Stanford Sentiment Treebank dataset](https://nlp.stanford.edu/sentiment/code.html)

## Usage

The easiest way to use the FastAPI HTTP endpoint is to download and run the following Docker image:
```
docker pull tommasobonomo/sentiment-classifier:inference
docker run --net=host tommasobonomo/sentiment-classifier:inference
```
This way the weights and configurations for the two proposed models are included directly in the Docker image, which is ready to run.

An HTTP endpoint will be available at [localhost:8000](http://localhost:8000), with readable docs at [localhost:8000/docs](http://localhost:8000/docs).

## Models and performance

### Baseline XGBoost + TfIdf
To extract features from the raw text, I implemented a Term-frequency inverse-document frequency algorithm ([TfIdf](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)) to build a vocabulary from the corpus of training sentences. 
I then applied a dimensionality reduction algorithm ([truncated SVD](https://scikit-learn.org/stable/modules/decomposition.html#lsa)) to reduce the dimensionality of the sentences encoded following the vocabulary. 
This preprocessing combination is usually known as Latent Semantic Analysis (LSA).

I then applied a standard XGBoost classifier, with some hyperparameter tuning that brought an increase in F1 score of around 0.05.

<details>
  <summary> XGBoost hyperparameter tuning </summary>

Through the Hydra package used to manage configurations in this repository, it is possible to run a hyperparameter sweep on a series of parameters. 
Below I reported the parameters and intervals that I optimized, through the [Ax Sweeper plugin](https://hydra.cc/docs/plugins/ax_sweeper/) for Hydra.

```
python -m scripts.fit_and_evaluate --multirun hydra/sweeper=ax \
  'xgboost_config.n_estimators=int(interval(5, 100))' \
  'xgboost_config.max_depth=int(interval(1, 10))' \
  'tfidf_config.output_dims=int(interval(5, 100))' \
  'tfidf_config.max_ngram_range=int(interval(1, 3))'
```
and the final best hyperparameters reported by the Bayesian Optimization algorithm:
```
{
    'xgboost_config.n_estimators': 95, 
    'xgboost_config.max_depth': 7, 
    'tfidf_config.output_dims': 49, 
    'tfidf_config.max_ngram_range': 2
}
```

</details>

### Transformer-based solution
I also implemented a transformer-based solution that uses a pre-trained Transformer encoder (in this case [DistilBERT](https://huggingface.co/distilbert-base-uncased)) with a classification head that can classify the whole given sentence.
I finetuned this architecture on the given Stanford Sentiment Treebank dataset, evaluating a few different hyperparameter choices on the `dev` split of the dataset.

## Performance on test split

The final metrics obtained on the `test` split are:

|           Model | F1-score | Accuracy | Precision | Recall |
|-----------------|----------|----------|-----------|--------|
| TfIdf + XGBoost |   0.6095 |   0.6033 |    0.5844 | 0.6370 |
|      DistilBERT |   0.8461 |   0.8541 |    0.8685 | 0.8248 |

where all metrics are considered in a binary classification scenario.
DistilBERT performs much better than the baseline.

## Data exploration

A brief data exploration notebook is provided in `notebooks/eda.ipynb`. It should be viewable as-is, but it could also be re-run if necessary.


