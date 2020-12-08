# MRC Stresstest

## Notebooks

the notebooks in the directory `notebooks` visualise and summarise
the results of the experiments and output figures and tables
used in the paper.

## Setup

```shell script
conda create -n submission-704-aaai python=3.7 anaconda
pip install dvc
conda install pytorch=1.6 torchvision cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
git submodule update
```
If running into problems with mysql, install mysql 
(e.g. `sudo apt-get install mysql` on ubuntu systems) 
([Reference](https://github.com/JudgeGirl/Judge-sender/issues/4#issuecomment-186542797))

We use `dvc` to for reproducibility.

## Evaluation
to reproduce the evaluation
```
./pull_all_predictions.sh
./pull_all_dev_results.sh
./pull_all_sam.sh
dvc repro evaluate-intervention-squad1  --downstream --force
dvc repro evaluate-intervention-hotpotqa  --downstream --force
dvc repro evaluate-intervention-drop  --downstream --force
dvc repro evaluate-intervention-newsqa  --downstream --force
```

## Getting the data
To pull *everything* (including the trained models, training and evaluation data), run
```shell script
./pull_all_datasets.sh
dvc pull
```
(ignore the errors)
## Training
to reproduce the training of any model
```
./pull_all_datasets.sh
dvc repro train-{model}-on-{dataset} --force
```
where model of `bidaf, 
bert-base-uncased, bert-large-uncased, roberta-base, roberta-large,
albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2, 
t5-small, t5-base, t5-large` and dataset is one of `squad1, hotpotqa,
 newsqa, drop`, e.g.
```
dvc repro train-bert-base-uncased-on-squad1 --force
```
Beware that the code is configured to run on 4 GPUs with 16 GB RAM
and [FP16](https://github.com/nvidia/apex) training.
If any of those do not work with your system,  you will need to 
adapt the corresponding command (in this case 
`train-bert-base-uncased-on-squad1`)
in `dvc.yaml` and reduce the batch
size, remove the `--fp16` flag or whatever it is that is not 
working for you.

## Annotations
To view the annotations use brat and import the files in
`data/brat-data-annotated`

## Generation
To generate your own stresstest version run
```
python main.py generate-balanced \
 --config conf/evaluate.json \
 --seed 1337 \
 --num-workers 8 \
 --do-save \
 --out-path $YOUR_OUT_PATH \
 --multiplier 2
```
Where the ratio of question types is defined in `conf/evaluate.json`, e.g.
```hocon
{
  "num_modifiers": 3, # number of sam, 1...num_modifiers
  "reasoning_map": { # ratio of question types
    "retrieval": 10,
    "retrieval_reverse": 10,
    "retrieval_two": 2,
    "retrieval_two_reverse": 2,
    "bridge": 2,
    "bridge_reverse": 2,
    "comparison": 1,
    "comparison_reverse": 1,
    "argmax": 5,
    "argmin": 5
  },
  "world": {
    "num_sentences": 6,
    "num_players": 12
  },
  "domain": "football",
  "modify_event_type": "goal",
  "unique_actors": true
}
```
and `multiplier` defines the scaling of this ratio.
The total number will be `num_modifiers * sum(v for v in reasoning_map.values()) * multiplier`
