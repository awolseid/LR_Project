## LR4WomenWealthIndex End2End Implementation
###

```
Project/
├── data/
│   ├── raw/                                # Folder to save raw data
│   ├── train_test/                         # Folder to save train and test data
│   ├── transformer/                        # Folder to save pickle file of transformer
├── logs/                                   # Folder to save log files
│   ├── mm_dd_HH_MM_SS.log                  # Log file example
├── models/
│   ├── trained_models/                     # Folder to save pickle file of trained models
│   ├── train_test_scores.csv               # Train and test performance scores
├── notebooks/                              # Folder of EDA notebook and original data
├── src/
│   ├── components/
│   │   ├── data_loader.py                  # Script for loading data
│   │   ├── data_cleaner.py                 # Script for cleaning data
│   │   ├── data_transformer.py             # Script for transforming data
│   │   ├── model_trainer.py                # Script for training model
│   │   ├── __init__.py
│   ├── utils.py                            # Script for utility functions
│   ├── logger.py                           # Script for logs
│   ├── exception.py                        # Script for custom exceptions
│   ├── __init__.py
├── prediction/
│   ├── templates/                          # Templates for model deployment
|   ├── prediction.py                       # Script for making prediction 
│   ├── __init__.py
├── train.py                                # Script for training model
├── deploy.py                               # Script for deploying trained model using Flask
├── requirements.txt                        # List of dependencies
├── setup.py                                # Setup script for packaging
├── .gitignore
└── README.md

```