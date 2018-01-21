# sklearn-exp
Experimental codes with sklearn, XGboost and LightGBM


## Commands to make|create envs
conda install -c conda-forge lightgbm 
conda install -c anaconda py-xgboost

conda env export | grep -v "^prefix: " > environment.yml
conda env create -f environment.yml


