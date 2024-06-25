# Sentence Reordering using Transformer architecture

This task was part of my Deep Learning examination on june 2024 for the Master’s program in Artificial Intelligence at the University of Bologna. The objective of this project create a mode capable of reordering the words inside a sentece. The input is presented as a sequence of words in a random permutation and the task it to recostruct the original English sentence.

In this study I suggest a Transformer Seq to Seq model capable of generating rearranged sentences that maintain the original sentences’ meaning and grammatical correctness.


## Dataset 
The Dataset was taken from HuggingFace and contains a large (3.5M+ sentence) knowledge base of generic sentences. The HuggingFace page of the dataset can be found [here](https://huggingface.co/datasets/community-datasets/generics_kb)


## Evaluation Metric

Given the original string and the prediction the evaluation function: 

- check for the longest matching sequence between the original and the predicted string ()
- divide the lenght of the longest matching sequence by the longest lenght between the predicted and the original sentence    

$$ \frac{ LongestMatchingSequence(original, prediction)}{max(len(original), len(prediction))} $$ 

#### Example

```
original = "at first henry wanted to be friends with the king of france"
generated = "henry wanted to be friends with king of france at the first"

print("your score is ", score(original, generated))
```
```
$ your score is  0.5423728813559322
```

## Constraints 
- No pretrained model can be used 
- The neural network model should have less than 20M parameters
- No postprocessing techniques can be used

## Model and hyperparameters 
For this task it was implemented a Transformer Seq to Seq model.

This kind of model si composed by 2 main parts:

- Encoder: read the input sequence (in this case the shuffled words) and produces a fixed-dimensional vector representation.
- Decoder: generate the output sequence (original sentence) from the input given by the Encoder.

This kind of models are well known and largely used in natural language tasks (NLP) as may be translations, summarization and classifications.

### Why this model? 
The main reason to choose this kind of architecture is the self-attention mechanism. This characteristic should help the model capture the semantic meaning of the words, helping it achive good performance in reorder the words inside a phrase

## Hyperparameters
List of the hyperparameters used: 

**Model (~8M param)**: 
- `EMBEDDING_DIM` = 128
- `LATENT_DIM` = 512
- `NUM_HEADS` = 20 

\
**Training**: 
- `EPOCH` = 30
- `BATCH_SIZE` = 256

## Results
The final model has a score of `~0.49` using the provided evaluation metrics, way above the estimated performance of a random classifier, estimated to be around ~0.19 with a standard deviation of ~0.06. 

## Excecution Enviroment
The notebook has been created working on a [Kaggle notebook](https://www.kaggle.com/code). A requirement.txt file is provided inside the repo and report all the dependences found at the end of the excecution of the Kaggle notebook. 
The same code can also be excecuted in colab, but the dependences may not work. 