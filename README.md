RAG-based Dynamic Prompting for Few-Shot Biomedical NER

This repository provides an implementation of the dynamic prompting approach based on Retrieval-Augmented Generation (RAG) for few-shot biomedical Named Entity Recognition (NER). The code demonstrates the transition from static to dynamic prompting strategies, leveraging retrieval engines to enhance model adaptability in data-sparse scenarios.

Dataset Example: Reddit_Impacts Dataset

The repository includes an example using the Reddit Impacts dataset, which focuses on extracting clinical and social impact entities from text related to opioid use.
Features

    Static Prompting: Includes task-specific static prompts with components like dataset descriptions, entity definitions, and high-frequency instances.
    Dynamic Prompting with RAG: Dynamically retrieves annotated examples based on contextual similarity using retrieval engines (e.g., TF-IDF, SBERT, ColBERT).

Code Structure

    Retrieval Engines/: Includes retrieval engines such as TF-IDF, SBERT, DPR, LSA and ColBERT.
    data/: Example dataset (Reddit_Impacts) and preprocessed files.
    preprocessing/: preprocessing codes for futher used to retrieve similar sentences.
    LLMs/:
    Evaluation/: golden set of Reddit_Impacts dataset and evaluation metrics.


    
