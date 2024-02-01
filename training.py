import copy
import torch
from generate_graphs import create_dataset
from nets import TemporalGNN


def train_test_split(dataset, train_ratio: float = 0.8):
    spanshot_count = len(dataset)
    train_snapshots = int(train_ratio * spanshot_count)
    
    train_iterator = dataset[0:train_snapshots]
    test_iterator = dataset[train_snapshots:]

    return train_iterator, test_iterator


def train(model, optimizer, train_dataset, eval_dataset, epochs):
    model.train()
    
    best_cost = float('inf')
    model_state_dict = None

    for epoch in range(epochs):
        loss = 0
        step = 0
        for i, snapshot in enumerate(train_dataset):
            snapshot = snapshot.to(device)

            # Get model predictions
            preds = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            
            loss += torch.sqrt(torch.mean((preds-snapshot.y)**2))
            step += 1

        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

            cost = evaluate(model, eval_dataset)
            if cost < best_cost:
                best_cost = cost
                model_state_dict = copy.deepcopy(model.state_dict())
    
    return model_state_dict, best_cost

@torch.no_grad()
def evaluate(model, dataset):
    model.eval()

    cost = 0
    cost_per_prediction = torch.empty(dataset[0].y.shape[1]).cuda()
    for time, snapshot in enumerate(dataset):
        snapshot = snapshot.to(device)
        preds = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost += torch.sqrt(torch.mean((preds-snapshot.y)**2))

        for j in range(preds.shape[-1]):
            cost_per_prediction[j]+=torch.sqrt(torch.mean((preds[:, j]-snapshot.y[:, j])**2))

    cost = cost / (time+1)
    cost_per_prediction = cost_per_prediction / (time + 1)

    cost = cost.item()
    print("Test MSE: {:.4f}".format(cost))

    print(f"Test: (per prediction): {[round(c.item(), 4) for c in cost_per_prediction]}")

    return cost

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_PYG_DATA = True

    # model/train paramas
    node_features = 2
    hidden_dim = 32
    epochs = 300
    learning_rate = 0.01

    # dataset creation params
    graphs_filename = 'graphs'
    input_width = 12
    shift = 12
    label_width = 12
    feature_names = ['frequency', 'degree']
    label_name = 'frequency'
    filter_nodes = True

    graphs_filepath = './graphs'

    dataset = create_dataset(graphs_filepath, input_width, shift, label_width, 
                                feature_names, label_name, normalize_features=False,
                                filter_nodes=filter_nodes)
    
    # # OPTIONAL: save dataset  
    if SAVE_PYG_DATA:
        dataset = torch.load('./pyg_data.pt')
 
    # split dataset to training and test sets
    train_dataset, test_dataset = train_test_split(dataset, train_ratio=0.9)
    
    # initialize model
    model = TemporalGNN(in_feats=node_features, hidden_dim=hidden_dim, 
                        periods=label_width).to(device)
    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    
    # train model
    model_state_dict, best_cost = train(model, optimizer, train_dataset, 
                                        test_dataset, epochs=epochs)

    # print and save best model parameters
    print(f"Final best cost {best_cost:.4f}")
    torch.save(model_state_dict, 'model_state_dict.pt')
