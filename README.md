# RAG-based Dynamic Prompting for Few-Shot Biomedical NER

This repository provides an implementation of the dynamic prompting approach based on Retrieval-Augmented Generation (RAG) for few-shot biomedical Named Entity Recognition (NER). The code demonstrates the transition from static to dynamic prompting strategies, leveraging retrieval engines to enhance model adaptability in data-sparse scenarios.

## Dataset Example: Reddit_Impacts Dataset

The repository includes an example using the Reddit Impacts dataset, which focuses on extracting clinical and social impact entities from text related to opioid use.

## Features

   **Static Prompting:** Includes task-specific static prompts with components like dataset descriptions, entity definitions, and high-frequency instances.
    
   **Dynamic Prompting with RAG:** Dynamically retrieves annotated examples based on contextual similarity using retrieval engines (e.g., TF-IDF, SBERT, ColBERT).

## Code Structure

   **Retrieval Engines/:** Includes retrieval engines such as TF-IDF, SBERT, DPR, LSA and ColBERT.
    
   **data/:** Example dataset (Reddit_Impacts) and preprocessed files.
    
   **preprocessing/:** preprocessing codes for futher used for retrieving similar sentences.
    
   **LLMs/:** This folder contains scripts for interacting with large language models (LLMs) like GPT-4 and LLaMA 3. The code is designed to:

      1. Access APIs: Establish connections to model APIs (e.g., OpenAI API for GPT-4, local API for LLaMA 3).
      2. Provide Prompts: Dynamically or statically generate and send prompts to the LLMs.
      3. Retrieve Predictions: Process the responses from the LLMs, format them as needed, and prepare the outputs for evaluation.
    
   **Evaluation/:** golden set of Reddit_Impacts dataset and evaluation metrics.

## Quick Start

  **Prepare Dataset:** The dataset used in this project is located in the data/ folder. To ensure compatibility with the retrieval stage, scripts for converting the data into the required format are provided in the preprocessing/ folder.

  **Indexing and Retrieval:** 
  
    1. The file RAG_for_Reddit_Impacts_Dataset.ipynb provides code for performing retrieval using SBERT, TF-IDF, LSA, and DPR.
    2. The file ColBERT_for_Reddit_Impacts_Dataset.ipynb contains code specifically for retrieval using ColBERT.
  
  **Dynamic Prompting:** Use the scripts in LLMs/ to access GPT-4 and Llama 3 for predictions.

  **Evaluation** Use the evaluation metrics and golden set in Evaluation/ to calculate F1-score.


    
