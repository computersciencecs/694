# RecRankerEval: A Reproducible Framework for Deploying and Evaluating LLM-based Top-k Recommenders

This repository hosts the code of RecRankerEval. 
This project builds upon and extends components from SELFRec and RecRanker. 
We sincerely thank the authors for their open-source contributions, which greatly facilitated our work.

## 1. Install
* Install environment

We implement RecRankerEval in Python Version 3.10.13, and PyTorch Version 2.4.0+cu121.

./RecRankerEval/requirements.txt shows the environment in which the experiments are run.

Install the environment need to first pull the following Docker image:

```bash
docker pull reconmmendationsystem/notebook:cuda12.1_unsloth
```

Since some packages need to be compatible with each other and will overlap during installation, the following additional packages need to be installed in order:

```bash
pip install vllm==0.6.2

pip install transformers==4.45.0

pip install https://download.pytorch.org/whl/cu121/torch-2.4.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=28bfba084dca52a06c465d7ad0f3cc372c35fc503f3eab881cc17a5fd82914e7

pip install flash-attn==2.6.3

pip install accelerate==1.0.1

pip install peft==0.13.2

pip install jsonlines
```

## 2. Quick Start

Run main.py in the ./RecRankerEval/train-and-inference/ directory:

#1 Instruction tuning Llama2 and inference
```python
python main.py \
  --task_type fine_tune \
  --model_type meta-llama/Llama-2-7b-hf \
  --training_type pointwise \
  --token  \
  --epochs 3 \
  --batch_size 2 \
  --grad_acc_steps 64 \
  --lr 2e-5
```

#2 Instruction tuning Llama3 and inference
```python
python main.py \
  --task_type fine_tune \
  --model_type meta-llama/Llama-3.1-8B-Instruct \
  --training_type pointwise \
  --token  \
  --epochs 3 \
  --batch_size 2 \
  --grad_acc_steps 64 \
  --lr 2e-5
```

#3 zero_shot for Llama2
```python
python main.py \
  --task_type zero_shot \
  --token  \
  --model_type meta-llama/Llama-2-7b-hf \
  --training_type pointwise \
  --inference_batch_size 80 \
  --tensor_parallel_size 1
```

#4 load checkpoint, instruction tuning Llama3 and inference
```python
python main.py \
     --task_type fine_tune \
     --model_type meta-llama/Llama-2-7b-hf \
     --training_type pointwise \
     --token  \
     --adapter_dir ./checkpoint \
     --skip_train
```

#5 use gpt to inference
```python
python main.py \
  --task_type zero_shot \
  --openai_api_key  \
  --model_type gpt-3.5-turbo-0125 \
  --training_type pointwise
```

## 3. Process the original dataset

* Load original dataset

All the original rating.csv files and item.csv files are in the ./RecRankerEval/dataset directory.

* Preprocessing the dataset

ML-100K and ML-1M can be used directly, while BookCrossing and AmazonMusic need preprocessing.

In ./RecRankerEval/dataprocess/bookcrossing, use bookcrossing-data-process.ipynb to process the original data. 
The input files are the ratings file book-crossing.csv and item information file book-crossing.item.csv decompressed in ./RecRankerEval/dataset/bookcrossing. 
The output files are the processed bookcross8-4.csv and book-crossing.item-12.csv.

In ./RecRankerEval/dataprocess/amazonmusic, use amazonmusic-data-process.ipynb to process the original data.
The input files are the rating file inter.csv and item information file item.csv decompressed in ./RecRankerEval/dataset/amazonmusic.
The output files are the processed inter2-5.csv and itemnew-processed.csv.

* Split the dataset

Use ./RecRankerEval/dataprocess/splitdataset.py in each dataset folder to split the preprocessed rating files into datasets.
The input files are the preprocessed rating files in each dataset folder in ./RecRankerEval/dataset/.
The output files are like.txt, dislike.txt, train_set.txt, valid_set.txt, and test_set.txt.
The output files are saved in the ./RecRankerEval/dataset/ directory corresponding to the dataset.

## 4. Run the initial recommendation model

In order to obtain the pkl file required to build prompts as well as the initial recommendation list and ground truth, RecRankerEval needs to run the initial recommendation model first.

### Processing Format

Use ./RecRankerEval/dataprocess/processing format.ipynb to modify the dislike.txt of the corresponding dataset to a file with spaces as delimiters to adapt to the initial recommendation model.
The input file is dislike.txt of the corresponding dataset.
The output file is dislike_set.txt of the corresponding dataset.

### Quick Start

Run ./RecRankerEval/top-k-recommendation.py to use different initial recommendation models to obtain the relevant files needed to build prompts later.
The input files are train_set.txt, test_set.txt, valid_set.txt, and dislike_set.txt in the corresponding dataset files in the ./RecRankerEval/dataset/ directory.
The output files are the initial recommendation list file Modelrec_save_dict.csv, the ground truth file Modelgt_save_dict.csv in the model_result subfolder of the corresponding dataset file in the ./RecRankerEval/dataset/ directory, and user.pkl, user_id_mapping.pkl, rating_matrix.pkl, pred.pkl, item.pkl, item_id_mapping.pkl, and item_id_mapping-all.pkl in the ./RecRankerEval/dataset/ directory.

The parameters and configuration files of the model are in the files corresponding to the model name in the ./RecRankerEval/conf directory.

* Configure the first dimension of RecRankerEval - initial recommendation model

RecRankerEval supports the use of different initial recommendation models.
The name of the initial recommendation model can be modified in the running code, for example:

```python
!python top-k-recommendation.py --model LightGCN
```

RecRankerEval can enable users to extend and replace the initial recommendation model for use.
To change the initial recommendation model, follow the steps below:
Save the initial recommendation model in the ./RecRankerEval/model directory, and add the code mentioned above to generate and save csv and pkl files according to the existing model files in RecRankerEval;
Create a conf configuration file corresponding to the new model name in the ./RecRankerEval/conf directory;
Add the new model name in the top-k-recommendation.py file.

* Configure the second dimension of RecRankerEval - dataset

In addition, RecRankerEval supports users to change the dataset to run the initial recommendation model:
If it is an existing dataset, the user can directly modify the name of the dataset in the running code, for example:

```python
!python top-k-recommendation.py --dataset ml-1m
```

If the user wants to add additional datasets, follow the steps below:
Replace the dataset name in the corresponding initial recommendation model conf file in the ./RecRankerEval/conf directory;
Replace the dataset name in the top-k-recommendation.py file;
Create a new dataset in ./RecRankerEval/dataset, and create a model_result subfolder in the dataset.

## 5. Build train and test Prompts, Run Inference with the Instruction-Tuned LLM

### Organise input files for train prompts

The input files are: train_set.txt, dislike.txt, movie_info.csv, user.pkl, user_id_mapping.pkl, rating_matrix.pkl, pred.pkl, item.pkl, item_id_mapping.pkl, and item_id_mapping-all.pkl.
The output files are: pointwise.jsonl, pairwise.jsonl and listwise.jsonl.
Put all input files in the ./RecRankerEval/train-and-inference/ directory.

### Build training prompts

* Configure the third dimension of RecRankerEval - user sampling

Run make-train-1.py or make-train-db.py in the ./RecRankerEval/train-and-inference/ directory to generate pointwise.jsonl, pairwise.jsonl and listwise.jsonl for training respectively.
Users can change the sample_method in the file to implement different user sampling methods.

```python
python make-train-1.py --datasets ml-1m --sample-method kmeans
```

```python
python make-train-db.py --datasets ml-1m --sample-method db

```

### Organise input files for inference prompts

The input files are: train_set.txt, dislike.txt, movie_info.csv, user.pkl, user_id_mapping.pkl, rating_matrix.pkl, pred.pkl, item.pkl, item_id_mapping.pkl, item_id_mapping-all.pkl, item information file, Modelgt_save_dict.csv and Modelrec_save_dict.
The output files are: pointwisetest.jsonl, pairwisetest.jsonl and listwisetest.jsonl.
Put all input files in the ./RecRankerEval/train-and-inference/ directory.

### Build inference prompts

Run make-testprompt.py in the ./RecRankerEval/train-and-inference/ directory to pointwisetest.jsonl, pairwisetest.jsonl, pairwise_invtest.jsonl and listwisetest.jsonl for inference respectively.

```python
python make-testprompt.py --datasets ml-1m --models LightGCN --rec-list ./Light1rec_save_dict1.csv --gt-list ./Light1gt_save_dict1.csv
```

For the pairwise instruction tuning method, user needs to run ./RecRankerEval/dataprocess/merge-allpairwise-testprompt.py first to merge the two jsonl files for forward and reverse comparison.
The input is pairwisetest.jsonl and pairwise_invtest.jsonl, and the output is pairwiseall.jsonl.

For pointwise without data leakage, we provide two files to process the training and reasoning prompts of pointwise respectively.
./RecRankerEval/dataprocess/fix-pointwise-trainprompt.py is used to process the training prompt of pointwise.
./RecRankerEval/dataprocess/fix-pointwise-testprompt.py is used to process the inference prompt of pointwise.

Copy the previously generated training and test prompts to the dataset directory corresponding to ./RecRankerEval/train-and-inference, and then run the ./RecRankerEval/train-and-inference/main.py to instruction tuning LLM and then inference (as shown in the Quick Start).

The output is inference.txt.

```python
#1 Instruction tuning Llama2 and inference
!python main.py \
  --task_type fine_tune \
  --model_type meta-llama/Llama-2-7b-hf \
  --training_type pointwise \
  --token  \
  --epochs 3 \
  --batch_size 2 \
  --grad_acc_steps 64 \
  --lr 2e-5
```

* Configure the forth and fifth dimension of RecRankerEval - Instruction tuning method and LLM backbone

RecRankerEval supports using different LLM backbones and different instruction tuning method.
In the command of running main.py, change task_type to zero_shot to switch to zero shot learning; change model_type to switch to Llama3 or gpt3.5; change training_type to switch to listwsie or pointwise. Detailed configuration information is located in ./RecRankerEval/train-and-inference/config.py. Quick Start provides running examples of different configurations.

## 6. Process the results

After inference is completed, copy the output file inference.txt to the ./RecRankerEval/dataprocess/process-inference-results directory, and then run the ipynb script to process the data according to the corresponding ranking method.
We provide an example of processing results in the ./RecRankerEval/dataprocess/process-inference-results/example directory.

## Code Structure


```python
.
├── RecRankerEval
│   ├── dataprocess
│       ├── amazonmusic
│           └── amazonmusic-data-process.ipynb
│               └── splitdataset.py
│       ├── bookcrossing
│           └── bookcrossing-data-process.ipynb
│               └── splitdataset.py
│       ├── ml-100k
│           └── splitdataset.ipynb
│       ├── ml-1m
│           └── splitdataset.ipynb
│       ├── process-inference-results
│           ├── example
│               ├── gt-match-output.csv
│               ├── inference.rar
│               ├── pointwise-output-process.ipynb
│               └── pointwsietest.jsonl
│           ├── hybrid-output-process.ipynb
│           ├── listwise-output-process.ipynb
│           ├── pairwise-output-process.ipynb
│           └── pointwise-output-process.ipynb
│       ├── processing-format.ipynb
│       ├── merge-allpairwise-testprompt.py
│       ├── fix-pointwise-trainprompt.py
│       └── fix-pointwise-testprompt.py
│   ├── train-and-inference
│       ├── listwise
│           ├── merged_model
│           ├── results
│           ├── trained_model
│       ├── pointwise
│           ├── merged_model
│           ├── results
│           ├── trained_model
│           ├── checkpoint
│               ├── adapter_config.json
│               ├── adapter_model.safetensors
│               ├── training_args.bin
│           ├── pointwise.jsonl
│           └── pointwisetest.jsonl
│       ├── pairwise
│           ├── merged_model
│           ├── results
│           ├── trained_model
│       ├── config.py
│       ├── gptinference.py
│       ├── inference.py
│       ├── make-testprompt.py
│       ├── make-train-1.py
│       ├── make-train-db.py
│       ├── main.py
│       ├── train.py
│       └── zeroshot.py
│   ├── base
│       ├── graph_recommender.py
│       ├── recommender.py
│       ├── seq_recommender.py
│       ├── ssl_interface.py
│       ├── tf_interface.py
│       └── torch_interface.ipynb
│   ├── conf
│       ├── LightGCN.py
│       ├── MF.py
│       └── XSimGCL.py
│   ├── data
│       ├── augmentor.py
│       ├── data.py
│       ├── feature.py
│       ├── graph.py
│       ├── loader.py
│       ├── sequence.py
│       ├── social.py
│       └── ui_graph.py
│   ├── dataset
│       ├── amazonmusic
│           ├── dislike.txt
│           ├── inter.rar
│           ├── item.csv
│           ├── like.txt
│           ├── test_set.txt
│           └── train_set.txt
│       ├── bookcrossing
│           ├── dislike.txt
│           ├── book-crossing.item.rar
│           ├── book-crossing.rar
│           ├── like.txt
│           ├── test_set.txt
│           └── train_set.txt
│       ├── ml-100k
│           ├── dislike.txt
│           ├── dislike_set.txt
│           ├── movie_info.csv
│           ├── ratings.csv
│           ├── like.txt
│           ├── valid_set.txt
│           ├── test_set.txt
│           ├── train_set.txt
│           └── model_result
│       ├── ml-1m
│           ├── dislike.txt
│           ├── movie_info_ml1m.csv
│           ├── ratings.csv
│           ├── like.txt
│           ├── valid_set.txt
│           ├── test_set.txt
│           └── train_set.txt
│   ├── log
│   ├── model
│       ├── graph
│           ├── LightGCN.py
│           ├── MF.py
│           └── XSimGCL.py
│       └── sequential
│   ├── results
│   ├── util
│       ├── algorithm.py
│       ├── conf.py
│       ├── evaluation.py
│       ├── logger.py
│       ├── loss_tf.py
│       ├── loss_torch.py
│       ├── sampler.py
│       └── structure.py
│   ├── initialRec.py
│   ├── top-k-recommendation.py
│   └── requirements.txt
└── README.md
```
```
