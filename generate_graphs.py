import networkx as nx
import pickle
import csv
import math
from collections import defaultdict

import torch
from torch_geometric.data import Data
import torch.nn.functional as F

def load_data(filepath : str):
    with open(f'{filepath}.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)
    
def load_graphs(filepath : str):
    with open(f"{filepath}.pkl", 'rb') as f:
        G : nx.Graph() = pickle.load(f)
        return G

def create_data_buckets(data_filepath, bucket_length, shift):
    data = load_data(data_filepath)

    buckets = []
    grouped_data = {}
    for item in data:
        month = int(item['month'])
        keywords = [keyword.lower().strip() for keyword in item['authkeywords'].split('|') if item['authkeywords']]

        if month in grouped_data:
            grouped_data[month].append(keywords)
        else:
            grouped_data[month] = [keywords]

    count = 0
    while count < len(list(grouped_data.keys())):
        bucket = []

        for i in range(count + 1, count + 1 + bucket_length):
            try:
                bucket.extend(grouped_data[i])
            except:
                pass

        count+=shift
        buckets.append(bucket)

    return buckets


def create_graphs(data_buckets):
    graphs = []

    for count, data in enumerate(data_buckets):
        G = nx.Graph()
        for words in data:
            for i in range(len(words)):
                if not G.has_node(words[i]):
                    G.add_node(words[i], frequency=1)
                else:
                    G.nodes[words[i]]['frequency']+=1

            for i in range(len(words)):
                for j in range(i + 1, len(words)):                
                    if G.has_edge(words[i], words[j]) or G.has_edge(words[j], words[i]):
                        G.edges[words[i], words[j]]['co_occurrence']+=1
                    else:
                        G.add_edge(words[i], words[j], co_occurrence=1)
        
        for (node, val) in G.degree():
            G.nodes[node]['degree'] = val
        
        
        graphs.append(G)
    
    return graphs

def combine_features_and_labels(input_graphs, label_graphs, node_features, 
                                label, filter_nodes):

    G = nx.Graph()
    graph : nx.Graph
    for i, graph in enumerate(input_graphs):
        for (node_key, node_attr) in list(graph.nodes(data=True)):
            if not G.has_node(node_key):
                a = {k : [0] * len(input_graphs) for k in node_features}
                G.add_node(node_key, **a)
                G.nodes[node_key][f"y_{label}"] = [0]* len(label_graphs)
                
            for k in node_features:
                G.nodes[node_key][k][i] = node_attr[k]

        for (src, dest, edge_attr) in list(graph.edges(data=True)):
            k_l = edge_attr.keys()
            if not G.has_edge(src, dest):
                a = {k : [0] * len(input_graphs)  for k in k_l}
                G.add_edge(src, dest, **a)

            for k,v in edge_attr.items():
                G.edges[src, dest][k][i] = v

    for i, graph in enumerate(label_graphs):
        for (node_key, node_attr) in list(graph.nodes(data=True)):
            if G.has_node(node_key):
                G.nodes[node_key][f"y_{label}"][i] = node_attr[label]

    if filter_nodes:
        threshold = math.floor(len(input_graphs) / 3)
        for (node_key, node_attr) in list(G.nodes(data=True)):
            if any(n == 0 for n in node_attr[label][-3:]):
                G.remove_node(node_key)
                break

    return G

def create_data_objects(graphs, feature_names, label, 
                        input_width, label_width, normalize_features):
    
    edge_features = ['co_occurrence']
    
    dataset = []

    for CG in graphs:
        CG = CG.to_directed() if not nx.is_directed(CG) else CG

        data = defaultdict(list)

        number_of_nodes = CG.number_of_nodes()
        number_of_edges = CG.number_of_edges()
        graph_nodes = list(CG.nodes(data=True))
        graph_edges = CG.edges(data=True)

        # create edge_index
        mapping = dict(zip(CG.nodes(), range(number_of_nodes)))
        edge_index = torch.empty((2, number_of_edges), dtype=torch.long)
        for i, (src, dst) in enumerate(CG.edges()):
            edge_index[0, i] = mapping[src]
            edge_index[1, i] = mapping[dst]


        data['edge_index'] = edge_index.view(2, -1)
    
        X = torch.empty(number_of_nodes, len(feature_names), input_width)
        y = torch.empty(number_of_nodes, label_width)
        
        words = []
        for i, (node, node_attr) in enumerate(graph_nodes):
            X[i] = torch.stack([torch.tensor(node_attr[x]) for x in feature_names], dim=0)
            y[i] = torch.tensor(node_attr[f"y_{label}"])
            words.append(node)


        # edge features sum
        edge_attributes = torch.empty(number_of_edges)
        for i, (src, dest, edge_attr) in enumerate(graph_edges):
            edge_attributes[i] = torch.sum(torch.tensor(edge_attr[edge_features[0]]), dim=0)

        
        # Feature normalization
        if normalize_features:
            m = torch.max(X[:, 0], dim=-1)
            m2 = torch.max(X[:, 1], dim=-1)
            X[:, 0] = torch.div(X[:, 0], m.values.reshape(X.shape[0], 1))
            X[:, 1] = torch.div(X[:, 1], m2.values.reshape(X.shape[0], 1))
            edge_attributes = F.normalize(edge_attributes, dim=0)


        data['x'] = X
        data['edge_attr'] = edge_attributes
        data['y'] = y
        data['words'] = words
 
        data = Data.from_dict(data)

        dataset.append(data)

    return dataset

def create_dataset(graphs_filepath, input_width, shift, label_width, 
                   feature_names, label, normalize_features, filter_nodes):

    graphs = load_graphs(graphs_filepath)

    combined_graphs = []
    for i in range(0, len(graphs), shift):
        graph_start = graphs[i:i+input_width]
        graph_end = graphs[i+input_width:i+input_width+label_width]

        if len(graph_start) < input_width : break
        if len(graph_end) < label_width : break

        combined_graph = combine_features_and_labels(graph_start, graph_end, feature_names, 
                                                     label, filter_nodes)

        combined_graphs.append(combined_graph)

    pyg_data_objects = create_data_objects(combined_graphs, feature_names, label,
                                           input_width, label_width, normalize_features)

    return pyg_data_objects


if __name__ == "__main__":    
    data_filepath = './example_data'
    bucket_length = 1
    shift = 1

    # creates data buckets of size bucket_length
    data_buckets = create_data_buckets(data_filepath, bucket_length, shift)
    
    # creates graphs from data_buckets
    graphs = create_graphs(data_buckets)

    # saves graphs to pickle object
    with open('./graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)



