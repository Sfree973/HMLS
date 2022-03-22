##
Source codes for TKDE paper 《When Hardness and Hierarchy Make a Difference: Multi-Hop Knowledge Graph Reasoning in Few-Shot Scenarios》
## 
This open source code was published during the review of our article. 
Note that any falsification, plagiarism, and commercial use are not permitted until the article is accepted.

## Requirements

- python3 
- pytorch 

### Installation

``` bash
python3 -m pip install -r requirements.txt
```

## Data Preparation

Unpack the data files

``` bash
unzip data.zip
```

``` bash
# dataset FB15K-237
data/FB15K-237

# dataset NELL-995
data/NE
```

## Pretrain Knowledge Graph Embedding

``` bash
./experiment-emb.sh configs/<dataset>-<model>.sh --train <gpu-ID>
```


## Meta Learning

``` bash
./experiment-rs.sh configs/dataset-rs.sh --train <gpu-ID> --few_shot
```

## Fast Adaptation

``` bash
./experiment-rs.sh configs/dataset-rs.sh --train <gpu-ID> --adaptation --checkpoint_path model/dataset-point.rs.conve-xavier-n/a-200-200-3-0.001-0.3-0.1-0.5-400-0.02/checkpoint-<Epoch>.tar
```

## Test

``` bash
./experiment-rs.sh configs/dataset-rs.sh --inference <gpu-ID> --few_shot --checkpoint_path model/dataset-point.rs.conve-xavier-n/a-200-200-3-0.001-0.3-0.1-0.5-400-0.02/checkpoint-<Epoch_Adapt>-[relation].tar
```