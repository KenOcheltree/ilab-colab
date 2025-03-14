# Using InstructLab to add knowledge and skills to LLMs in a Jupyter notebook
### A guide for running InstructLab on Colab with a GPU
### By Kenneth Ocheltree, Steve Buckley

# Overview
This Jupyter notebook demonstrates InstructLab, an open source AI project that facilitates knowledge and skills contributions to Large Language Models (LLMs). InstructLab uses a novel synthetic data-based alignment tuning method for Large Language Models introduced in this [paper](https://arxiv.org/abs/2403.01081). The open source InstructLab repository available [here](https://github.com/instructlab/instructlab) provides additional documentation on using InstructLab.

The InstructLab method consists of three major components:
* **Taxonomy-driven data curation:**  The taxonomy is a set of training data curated by humans as examples of new knowledge and skills for the model.
* **Large-scale synthetic data generation:** A teacher model is used to generate new examples based on the seed training data. Since synthetic data can vary in quality, InstructLab adds an automated step to refine the example answers, ensuring they are grounded and safe.
* **Iterative model alignment tuning:** The model is retrained based on the synthetic data. The InstructLab method includes two tuning phases: knowledge tuning, followed by skill tuning.

<img src="./images/Flow.png" width="700">

The notebook demonstrates the open source version of InstructLab running on Colab with a GPU. It runs the InstructLab process from Python code without the need to specify detailed configuration parameters or type commands into a command line interface (CLI). This allows the user to focus on preparing input files and testing fine-tuned output models. It is broken into the following major sequential sections:
* Configuring InstructLab
* Training with InstructLab
* Inferencing with InstructLab

# Steps

## Step 1. Select a Colab Plan

1. Go to the the Colab page (https://colab.research.google.com/) and select Sign In at the upper right.


<img src="./images/ColabWelcome.png" width=700>

2. Select a Google Account to use with Colab. 

3. If you do not have a Google account, choose Sign in and on the Sign In page, select Create account at the lower right. Proceed though the steps to create a Google Account and then come back to the [Colab Sign In page](https://colab.research.google.com/) and sign in with your account

4. Once signed into Colab with your Google account, go to the [Colab Plan page](https://colab.research.google.com/signup?utm_source=notebook_settings&utm_medium=link&utm_campaign=premium_gpu_selector). 

<img src="./images/ColabPlan.png" width=700>

5. Select one of the four offered plans, since all give access to a GPU with memory of at least 18 GB. A typical single InstructLab run on a GPU will incur charges of one to two dollars. After selecting a plan, you will have to pay for it before you can use the GPU.

## Step 2. Place your Hugging Face Token in Colab
1. Go to the [Hugging Face site](https://huggingface.co/). If you have previously created a login with HuggingFace, select **Log In** and provide your credentials. Otherwise, select Sign Up and create a free HuggingFace account.

<img src="./images/HF.png" width=700>


2. Select your account icon in the upper right and choose **Access Tokens** from the dropdown menu.

<img src="./images/HFDropDown.png" width=700>

3. On the Access Token page, select Create Token at the upper right.

<img src="./images/HFAccessTokens.png" width=700>

4. Create a Fine-grained access token with the following permissions: “Read access to contents of all repos under your personal namespace”; “Read access to contents of all public gated repos you can access”. Save the token value for the next step. This is done on the Create new Access Token page, by selecting **Fine-grained**, name the Token "ilab", select the first two boxes under ***Repositories*** and move to the bottom of the page and select **Create Token**.

<img src="./images/HFCreateToken.png" width=700>

5. On the popup that shows your access token, select the Copy Button, then Done.

6. Now go back to [Colab](https://colab.research.google.com/) and select the key icon in the left column, as seen in the below image. On the new screen that appears as shown below, select **+Add new secret**. Name the secret "hf_token" and paste the copied HuggingFace Token into the "Value" field. Slide the Notebook Access slider to the right to enable it (it turns blue). Close the Secrets column when this is completed.

<img src="./images/ColabSecretEnabled.png" width=400>

## Step 3. Open the Jupyter Notebook in Colab

1. Go to https://colab.research.google.com/ and sign in with your Google account. 

2. Select File->Open notebook and choose Github on the left side. Enter “KenOcheltree” for the GitHub URL. Select the KenOcheltree/ilab-colab repository

<img src="./images/Github.png" width=600>

3. At this point, Select the notebook named running_instructlab_on_gpu.jpynb and it will open into Colab.

<img src="./images/RunningILab.png" width=700>

## Step 4. Select a GPU

1. From the top menu, select Runtime->Change runtime type. You now must select one of the GPU options L4 or A100 to run this notebook. The A100 GPU is a good option as it has more memory and runs the InstructLab Jupyter notebook faster. 

<img src="./images/ColabChangeRuntime.png" width=350>

2. You can monitor resource usage by selecting Runtime->View resources and change plans as needed.

**Note** This step can be performed whenever you want to change the GPU used. Changing the runtime will terminate the current session and lose current work. 

## Step 5. Run the first cell to perform pre-reset installs
1. Run the first code cell by clicking the arrow next to it. 

<img src="./images/ColabRunFirst.png" width=700>

2. When it completes it will ask you to restart the session. Restart the session when requested.

<img src="./images/ColabRestart.png" width=300>

## Step 6. Optionally, provide your own InstructLab QNA data set

You can optionally provide your own InstructLab QNA file for processing.

**Note:** You may want to run this notebook with an existing dataset before creating your own to understand the InstructLab flow.

Follow these steps to add your own dataset:
1. Create your own qna.yaml file following the directions on the InstructLab taxonomy [readme](https://github.com/instructlab/taxonomy).
1. Create a questions.txt file with related sample questions to use on inferencing.
1. Add your qna.yaml and sample questions.txt files to the /content/ilab/data/your_content_1 folder or the /content/ilab/data/your_content_2 folder by dragging and dropping them in the desired folder.
1. Double click on the /content/ilab/config.json file to edit and specify the qna_location where your data resides within the Dewey Decimal classification system. Close and save the config.json file.
1. You can now specify to run with your own data by selecting **Your Content 1** or **Your Content 2** in the next code cell.

<img src="./images/ColabFiles.png" width=800>

## Step 7. Run the second cell and select the InstructLab parameters

1. Run the second code code cell by clicking on the arrow next to it. Once the second cell is run, it presents a number of different parameters available for running InstructLab.

<img src="./images/IlabParms.png" width=800>

2. Select the desired run parameters. The parameters are as follows:
- Dataset - Choose between data on the 2024 Oscars, IBM Quantum technology, and Agentic AI
- Number of Questions and Answers (QNA) -Selects the number of QNAs to be synthetically generated. You may wish to select a small number of QNAs if you are introducing a new dataset.
- Number of Epochs: specifies the number of times the learning algorithm will work through the entire training dataset.
- Iterations - specifies the number of stages of training.
- Live Q&A - option to open an inferencing dialogue with untrained and trained models to compare results
- Download - option to download the created trained model

## Step 8. Run the remainder of the notebook

1. Select the third code cell, shown below, without running it, and click on **Runtime->Run cell and below** to run the rest of the notebook.

<img src="./images/ThirdCell.png" width=700>

2. The run will proceed as follows to configure InstructLab and create a model:
- Complete the environment setup by installing InstructLab and other required packages
- Configure the InstructLab installation
- Download LLM models for training and teaching
- Create a taxonomy with the new data
- Perform synthetic data generation
- Run model training with the synthetic data

3. After the model is created, you can choose to run inferencing on the model or download the trained model

- Optionally opens an inferencing dialogue with untrained and trained models
- Optionally downloads the trained model

After the run completes by exiting the inferencing loop, you can make another run if desired. To do that, go back to the second code cell, run it, and change the run parameters. Then select the third code cell without running it and click on Runtime->Run cell and below to run the rest of the notebook.

# Summary and next steps

This Tutorial demonstrated utilizing InstructLab for introducing datasets, data generation and model training. This notebook produced an InstructLab trained model that was available for inferencing and downloading.

Click on the Folder icon on the left to explore the files in the ilab folder. Preloaded QNA files and synthetically generated questions and answers can be found in this directory tree. You can also select **Runtime->View resources** to see the resource utilization of the run.

<img src="./images/ColabMore.png" width=1000>

Try running the notebook with the other example data sets. You can also experiment with how the number of Epochs and Iterations affect the quality of the generated model

After running through the InstructLab notebook, you are encouraged to create your own QNA file file as described in Step 6.

Once you have trained your model, you can download your model and experiment with it in other environments.

