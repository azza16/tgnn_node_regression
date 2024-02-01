## Repository for training a Temporal Graph Neural Network model for predicting frequency of keywords (node regression)

#### To train the model:
1. First run `python generate_graphs.py` to generate the graph data
    * This will create a *graphs.pkl* file
2. Then run `python training.py` to train the model using the generated graphs
    * This will save the best model parameters based on evaluation loss

You can evaluate the model and generate predictions using the `evalutation.ipynb` notebook
