#  WATER POTABILITY PREDICTION SERVICE
![image](https://github.com/dr-zaib/midterm-project/blob/main/image.jpeg)


## Overview 

This project stands as an evaluation test for the [midterm project]() from the [Machine Learning Zoomcamp course]() delivered by [DataTalks Club](https://datatalks.club/slack.html) conducted by Alex Grigorev.


#### Project structure

This project constains the following files: 

* [README.md](https://github.com/dr-zaib/midterm-project/blob/main/README.md): contains all the info about the project and how to access the service 
* [notebook.ipynb](https://github.com/dr-zaib/midterm-project/blob/main/notebook.ipynb): script with 
    - data preparation 
    - EDA, feature importance analysis: this analysis has been made genarally and also for each train model
    - model selection process and parameter tuning for each model

* [extra] [notebook_final.ipynb](https://github.com/dr-zaib/midterm-project/blob/main/notebook_final.ipynb): contains the final model only and a final training ad validation (cross-validation)

* [train.py](https://github.com/dr-zaib/midterm-project/blob/main/train.py):
    - training of the final model 
    - saving as [model_h2O_potability.bin](https://github.com/dr-zaib/midterm-project/blob/main/model_h2O_potability.bin)

* [predict.py](https://github.com/dr-zaib/midterm-project/blob/main/predict.py): 
    - loading the model
    - serving it via Google Cloud web service
* [Pipfile](https://github.com/dr-zaib/midterm-project/blob/main/Pipfile) and [Pipfile.lock](https://github.com/dr-zaib/midterm-project/blob/main/Pipfile.lock): files with dependencies

* [Dockerfile](https://github.com/dr-zaib/midterm-project/blob/main/Dockerfile): file containing the contairnerization info before deploying the service with GCP

* [testing.py](https://github.com/dr-zaib/midterm-project/blob/main/testing.py): useful script to test the service

## Problem description

Water potability has always been a "hot topic" in many African countries. Having access to drinking or even for other usage - treated water is not always automatic. 
If water sources may be largely available, resources to determine their how much recommandable they can be, are not. 
Having a service that may facilitate local biologists and researchers such as the service developed out of this [project](https://github.com/dr-zaib/midterm-project) could be a great and useful resource in the analysis and in the consequent potability rate estimation of the water given its characteristics: ph, hardness, solids (Total Dissolved Solids -TDS), chloramines, sulfate, conductivity, organic carbon, trihalomethanes and turbidity. 
Each of these characteristics are explained more in details [here](https://www.kaggle.com/datasets/adityakadiwal/water-potability) 
Furthermore, this service could also stand as an extra verification assest in the water usability controls context.

## Access the service 

To access the service just run the file [testing.py](https://github.com/dr-zaib/midterm-project/blob/main/testing.py) 
    - from your editor
    - from your terminal
In both cases make sure the environment you are running testing.py the python library [requests](https://pypi.org/project/requests/)


## Data 

* About Dataset 

*Context*

Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.

* Download the dataset [water_potability.csv](https://github.com/dr-zaib/midterm-project/blob/main/water_potability.csv) from this repository or from [kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability)


## Considerations

Different models had been trained for this project, but the RandomForestClassifier had revealed itself to be the best. 

- Logistic Regression: 
    
    tp:0, tn:415, fp:0, fn:240
    
    auc=0.524
    
    accuracy=0.634
    
    precision=0.0
   
    recall=0.0
    
    f1 score=0.0

- Decision Tree Classifier
    
    tp:75, tn:353, fp:62, 
    fn:165
    
    auc=0.613
    
    accuracy=0.653
    
    precision=0.547
    
    recall=0.312
    
    f1 score=0.398


- Random Forest Classifier
    tp:78,  tn:370,  fp:45, fn:162
    
    auc=0.663
    
    accuracy=0.684
    
    precision=0.634
    
    recall=0.325
    
    f1 score=0.43


- Xgboost Tree Classifier 
    tp:61,  tn:372,  fp:43,  fn:179
    
    auc=0.659
    
    accuracy=0.661
    
    precision=0.587
    
    recall=0.254
    
    f1 score=0.355



