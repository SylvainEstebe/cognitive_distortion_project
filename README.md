# 🪐 NLP Project: Data-Driven taxonomy for cognitive distortion

This repository holds the code for the project for Natural Language Processing (S2023). It contains the script of analysis.
You can use the interactive exploration of a cluster of negative thoughts on this dashboard: https://cognitive-distortion-d.onrender.com/

### ⏭ Workflows

![alt text]([https://github.com/[username]/[reponame]/blob/[branch]/image.jpg](https://github.com/SylvainEstebe/cognitive_distortion_project/tree/main/export#:~:text=NLP%20%2D%20Share%20(1).jpeg?raw=true)


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
3. Run the `analysis.ipybn` script to: 
    - see the analysis with k-mean and hdbscan clustering
    - re-export the data if needed
4. Run the `plot.ipybn` script to:
       - plot the different clusters


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

### 🗂 Bibliography
