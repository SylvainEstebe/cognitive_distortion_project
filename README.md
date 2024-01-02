# NLP Project: Data-Driven taxonomy for cognitive distortion

This repository holds the code for the project for Natural Language Processing (S2023). It contains the script of the analysis.

### ⏭ Workflows
![Texte alternatif](https://github.com/SylvainEstebe/cognitive_distortion_project/raw/main/export/NLP%20-%20Share%20(1).jpeg)

## Description of the dataset

| thought | original_label |
| ----------------- | -: |
| Someone I trusted stole something valuable of mine, I was extremely angry and wanted justice       | emotional reasoning |
| She doesn't respect me.        | overgeneralizing |
| **Total**         | **921**|

## Usage and reproducibility

The code was developed and tested on a MacBook Pro with macOS (Sonoma 14.1.2, python v3.9.6).

To reproduce the results , follow the steps below. All terminal commands should be run from the root directory of the repository.


1. Clone the repository
2. Create a virtual environment and install requirements
```
bash setup.sh
```
Think about replacing the different path with your own repertoire, which contains "reframing_dataset.csv" and "thinking_traps.jsonl"

3. Run the `preprocessing.ipybn` script to: 
    - merge the two dataset
    - some preprocessing
4. Run the `analysis.ipybn` script to: 
    - Do the embedding with sentence-transformer
    - reduction of dimension with UMAP
    - Clustering with k-mean and hdbscan
    - export of results
4. Run the `plot.ipybn` script to:
       - plot the different clusters
5. Run the `label.ipybn` script to:
       - generate label
6. Run the `dashboard.ipybn` script to:
       - generate dashboard


## Repository structure
```
├── code 
│   ├── analysis.ipynb
│   ├── clean.py
│   ├── clust.py
│   ├── dashboard.ipynb
│   ├── label.ipynb
│   ├── label.py
│   ├── plot.ipynb
│   ├── preprocessing.ipynb
│   ├── result.ipynb
├── env                                         <- Not included in repo
├── data
│   ├── corpus_disto.csv
│   ├── corpus_embedding.csv
│   ├── corpus_hdbscan_bayesian_optimisation.csv
│   ├── corpus_kmean.csv
│   ├── label_hdbscan_all-miniLM.csv
│   ├── label_hdbscan_roberta.csv
│   ├── reframing_dataset.csv
│   ├── thinking_traps.jsonl
├── doc                                   
│   ├──
│   ├── 
│   ├── 
│   └──
├── export   # all the image and html interactive plot                                 
├── .gitignore
├── README.md
├── dash_deploy.py # script used for the website dashboard
├── README.md
├── setup.sh 
├── requirements.txt
```
## Results
To display the results of the exploratory approach of cognitive distortion follow the following links:
- [dashboard](https://cognitive-distortion-d.onrender.com/)

These can also be found in the `export` folder of the repository.
