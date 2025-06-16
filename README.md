# RecRankerEval: A Reproducible Framework for Deploying and Evaluating LLM-based Top-k Recommenders

This repository hosts the code of RecRankerEval. 

## 1 – Install
* Install environment

We implement RecRankerEval in Python Version 3.10.13, and PyTorch Version 2.4.0+cu121.

./RecRankerEval/requirements.txt shows the environment in which the experiments are run.

Install the environment according to this file, or pull the following docker image and install additional packages:

```bash
docker pull reconmmendationsystem/notebook:cuda12.1_unsloth

pip install vllm==0.6.2

pip install transformers==4.45.0

pip install https://download.pytorch.org/whl/cu121/torch-2.4.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=28bfba084dca52a06c465d7ad0f3cc372c35fc503f3eab881cc17a5fd82914e7

pip install flash-attn==2.6.3

pip install accelerate==1.0.1

pip install peft==0.13.2
```

## 2 – Process the original dataset
* Load original dataset

All the original rating.csv files and item.csv files are in the ./RecRankerEval/dataset directory.

* Preprocessing the dataset

ML-100K and ML-1M can be used directly, while BookCrossing and AmazonMusic need preprocessing.

In ./RecRankerEval/1-dataprocess/bookcrossing, use 1-bookcrossing-data-process.ipynb to process the original data. 
The input files are the ratings file book-crossing.csv and item information file book-crossing.item.csv decompressed in ./RecRankerEval/dataset/bookcrossing. 
The output files are the processed bookcross8-4.csv and book-crossing.item-12.csv.

In ./RecRankerEval/1-dataprocess/amazonmusic, use 1-amazonmusic-data-process.ipynb to process the original data.
The input files are the rating file inter.csv and item information file item.csv decompressed in ./RecRankerEval/dataset/amazonmusic.
The output files are the processed inter2-5.csv and itemnew.csv.

* Split the dataset

Use ./RecRankerEval/1-dataprocess/2-splitdataset.py in each dataset folder to split the preprocessed rating files into datasets.
The input files are the preprocessed rating files in each dataset folder in ./RecRankerEval/dataset/.
The output files are like.txt, dislike.txt, train_set.txt, valid_set.txt, and test_set.txt.
The output files are saved in the ./RecRankerEval/dataset/ directory corresponding to the dataset.

## 3 – Run the initial recommendation model

In order to obtain the pkl file required to build prompts as well as the initial recommendation list and ground truth, RecRankerEval needs to run the initial recommendation model first.

### Processing Format

Use ./RecRankerEval/1-dataprocess/Processing Format.ipynb to modify the dislike.txt of the corresponding dataset to a file with spaces as delimiters to adapt to the initial recommendation model.
The input file is dislike.txt of the corresponding dataset.
The output file is dislike_set.txt of the corresponding dataset.

### Quick Start

Run ./RecRankerEval/top_k_recommendation.py to use different initial recommendation models to obtain the relevant files needed to build prompts later.
The input files are train_set.txt, test_set.txt, valid_set.txt, and dislike_set.txt in the corresponding dataset files in the ./RecRankerEval/dataset/ directory.
The output files are the initial recommendation list file Modelrec_save_dict.csv, the ground truth file Modelgt_save_dict.csv in the model_result subfolder of the corresponding dataset file in the ./RecRankerEval/dataset/ directory, and user.pkl, user_id_mapping.pkl, rating_matrix.pkl, pred.pkl, item.pkl, item_id_mapping.pkl, and item_id_mapping-all.pkl in the ./RecRankerEval/dataset/ directory.

The parameters and configuration files of the model are in the files corresponding to the model name in the ./RecRankerEval/conf directory.

* Configure the first dimension of RecRankerEval - initial recommendation model

RecRankerEval supports the use of different initial recommendation models.
The name of the initial recommendation model can be modified in the running code, for example:

```python
!python top_k_recommendation.py --model MF
```

RecRankerEval can enable users to extend and replace the initial recommendation model for use.
To change the initial recommendation model, follow the steps below:
Save the initial recommendation model in the ./RecRankerEval/model directory, and add the code mentioned above to generate and save csv and pkl files according to the existing model files in RecRankerEval;
Create a conf configuration file corresponding to the new model name in the ./RecRankerEval/conf directory;
Add the new model name in the top_k_recommendation.py file.

* Configure the second dimension of RecRankerEval - dataset

In addition, RecRankerEval supports users to change the dataset to run the initial recommendation model:
If it is an existing dataset, the user can directly modify the name of the dataset in the running code, for example:

```python
!python top_k_recommendation.py --dataset ml-100k
```

If the user wants to add additional datasets, follow the steps below:
Replace the dataset name in the corresponding initial recommendation model conf file in the ./RecRankerEval/conf directory;
Replace the dataset name in the top_k_recommendation.py file;
Create a new dataset in ./RecRankerEval/dataset, and create a model_result subfolder in the dataset.

## 4 – Build train prompts and train

### Organise input files for train prompts

The input files are: train_set.txt, dislike.txt, movie_info.csv, user.pkl, user_id_mapping.pkl, rating_matrix.pkl, pred.pkl, item.pkl, item_id_mapping.pkl, and item_id_mapping-all.pkl.
The output files are: pointwise.jsonl, pairwise.jsonl and listwise.jsonl.
Put all input files in the ./RecRankerEval/1-dataprocess/ directory.

### Build training prompts

* Configure the third dimension of RecRankerEval - user sampling

Run make-train-random.py or make-trainprompt-db.py in the ./RecRankerEval/1-dataprocess/ directory to generate pointwise.jsonl, pairwise.jsonl and listwise.jsonl for training respectively.
Users can change the sample_method in the file to implement different user sampling methods.

### Train the LLM
In the ./RecRankerEval/2-train-and-inference directory, copy train.py to the corresponding ranking method folder for training.
Copy the training jsonl file of the ranking method generated in the ./RecRankerEval/1-dataprocess/ directory to the corresponding ranking method folder in the ./RecRankerEval/2-train-and-inference directory as input.
Set OpenAI API key in train.py: os.environ["HF_TOKEN"] = " ".
Run train.py.

```python
!python train.py \
  --token  \
  --model_name meta-llama/Llama-2-7b-hf \
  --data_path ./listwise.jsonl \
  --output_dir ./results \
  --epochs 3 \
  --batch_size 2 \
  --grad_acc_steps 64 \
  --lr 2e-5
```

* Configure the forth dimension of RecRankerEval - LLM backbone
RecRankerEval supports using different LLM backbones.
If the user uses an additional fine-tunable LLM, the model_name can be switched to meta-llama/Llama-3.1-8B-Instruct in the training command.
If the user uses non-fine-tunable gpt or zero shot, the user can copy gptinference.py or zerotest.py in the ./RecRankerEval/1-dataprocess/ directory to the corresponding instruction adjustment folder in the ./RecRankerEval/2-train-and-inference directory, set the OpenAI API key and run.

## 5 – Build inference prompts and inference

### Organise input files for inference prompts

The input files are: train_set.txt, dislike.txt, movie_info.csv, user.pkl, user_id_mapping.pkl, rating_matrix.pkl, pred.pkl, item.pkl, item_id_mapping.pkl, item_id_mapping-all.pkl, Modelgt_save_dict.csv and Modelrec_save_dict.
The output files are: pointwisetest.jsonl, pairwisetest.jsonl and listwisetest.jsonl.
Put all input files in the ./RecRankerEval/1-dataprocess/ directory.

### Build inference prompts

Run make-testprompt in the ./RecRankerEval/1-dataprocess/ directory to pointwisetest.jsonl, pairwisetest.jsonl, pairwise_invtest.jsonl and listwisetest.jsonl for inference respectively.

### Inference the LLM

* Configure the fifth dimension of RecRankerEval - instruction tuning method

In the ./RecRankerEval/2-train-and-inference directory, copy inference.py to the corresponding ranking method folder for inference.
Copy the inference jsonl file of the ranking method generated in the ./RecRankerEval/1-dataprocess/ directory to the corresponding ranking method folder in the ./RecRankerEval/2-train-and-inference directory as input.
Among them, for the pairwise instruction adjustment method, you need to run ./RecRankerEval/1-dataprocess/Merge-allpairwise-testprompt.py first to merge the two jsonl files for forward and reverse comparison.
The input is pairwisetest.jsonl and pairwise_invtest.jsonl, and the output is pairwiseall.jsonl.
Run inference.py.
The output is inference.txt.

## 6 – Process the results

After inference is completed, copy the output file inference.txt to the ./RecRankerEval/1-dataprocess/resultsprocess directory, and then run the ipynb script to process the data according to the corresponding ranking method.

## Code Structure

```python
.
├── RecRankerEval
│   ├── 1-dataprocess
│       ├── amazonmusic
│           └── 1-amazonmusic-data-process.ipynb
│               └── 2-splitdataset.py
│       ├── bookcrossing
│           └── 1-bookcrossing-data-process.ipynb
│               └── 2-splitdataset.py
│       ├── ml-100k
│           └── 1-splitdataset.ipynb
│       ├── ml-1m
│           └── 1-splitdataset.ipynb
│       ├── resultsprocess
│           ├── hybrid-output-process.ipynb
│           ├── listwise-output-process.ipynb
│           ├── pairwise-output-process.ipynb
│           └── pointwise-output-process.ipynb
│       ├── Processing Format.ipynb
│       ├── gptinference.py
│       ├── inference.py
│       ├── make-testprompt.py
│       ├── make-train-random.py
│       ├── make-trainprompt-db.py
│       ├── Merge-allpairwise-testprompt.py
│       ├── fix-pointwise-trainprompt.py
│       ├── fix-pointwise-testprompt.py
│       ├── train.py
│       └── zerotest.py
│   ├── 2-train-and-inference
│       ├── listwise
│           ├── merged_model
│           ├── results
│           ├── trained_model
│       ├── pointwise
│           ├── merged_model
│           ├── results
│           ├── trained_model
│       ├── pairwise
│           ├── merged_model
│           ├── results
│           ├── trained_model
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
│   ├── SELFRec.py
│   ├── top_k_recommendation.py
│   └── requirements.txt
```
```
