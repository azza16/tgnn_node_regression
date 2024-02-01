## Repository for training a Temporal Graph Neural Network model for predicting frequency of keywords (node regression)

#### Task description:
* Based on .csv data containing keywords appearing together in scientific journals for specific months of an 8-year period (2014-2021), train model to predict the future frequency (e.g. for the next year) of the same keywords
* The main steps involve:
    1. Creating data windows of one year (8 data windows in total)
    2. Creating graphs for each data window in order to leverage structural dependencies between keywords (e.g. co-occurence)
    3. Transforming graphs into proper format in order to be ingested by Temporal GNN model
    4. Training the model

#### To train the model:
1. First run `python generate_graphs.py` to generate the graph data
    * This will create a *graphs.pkl* file
2. Then run `python training.py` to train the model using the generated graphs
    * This will save the best model parameters based on evaluation loss

You can evaluate the model and generate predictions using the `evalutation.ipynb` notebook
