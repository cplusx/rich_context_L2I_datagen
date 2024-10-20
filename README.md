The dataset generation code for preparing the synthetic training data in [Rethinking The Training And Evaluation of Rich-Context Layout-to-Image Generation (NeurIPS 2024)](https://www.arxiv.org/pdf/2409.04847)

This repository shows the data preparation for CC3M dataset. Other source of data can be prepared in the same protocal.

### 1. Download images
We use [`img2dataset`](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) to download the CC3M dataset. 

Each downloaded subfile contains 10,000 downloaded images. The files are in `.tar` format, we extract each tar file into a subfolder

### 2. Initialize the environment
The data generation pipeline requires several pre-trained models, you can create a new conda environment and install these models by running `bash init_env.sh`

Note: the script install the default pytorch version, if your hardware requires specific pytorch version, please edit the installation accordingly.

### 3. Generate the meta file for each process batch
Run `prepare_train_json_cc3m.py` to prepare the meta file for each subfolder. 

Before you call the script, please ensure to change the CC3M path and save dir in the script
```
data_dir = '/home/ubuntu/CC3M/cc3m-images'
save_dir = 'cc3m_process_json_768'
```

After running the script, you should get a folder containing many meta files like:
```
[
    {
        "image": image_path1,
        "caption": image_caption1
    },
    {
        "image": image_path2,
        "caption": image_caption2
    },
    ...
]
```

### 4. Start Generation

You can run script `bash run_generation_cc3m.sh` to start the generation. Before you start, please ensure to modify the image folder and meta file folder accordingly. You may also want to change the number of meta files to process by changing the `folder_idx_start` and `folder_idx_end`.

This script loops over all GPUs to find an idle GPU for the generation task. If you do not want this feature, please refer to the `run_generation()` function in the bash file for launching a single generation task.


## Acknowledgement

Some of the code used in this work is adapted from [InstanceDiffusion](https://github.com/frank-xwang/InstanceDiffusion/tree/main), we appreciate the contributions of above authors.