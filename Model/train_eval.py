from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import  BatchNorm1d
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader, ImbalancedSampler
from torch_geometric.nn import  TopKPooling,  GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import torch.optim as optim
import os


LR = 0.001
STEP_SIZE = 8
EPOCHS = 50

y_pred = []
y_true = []

class NEW(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        return ['edge.csv', 'graphid2label.csv', 'node2graphID.csv', 'nodeattrs.csv']

    @property
    def processed_file_names(self):

        return 'data.pt'

    def download(self):
        pass

    def process(self):

        path = os.path.join(self.raw_dir, 'nodeattrs.csv')
        node_attrs = pd.read_csv(path, sep=',', header=0, index_col=0)
        node_attrs.index += 1


        path = os.path.join(self.raw_dir, 'edge.csv')
        edge_index = pd.read_csv(path, sep=',', header=0)
        edge_index.index += 1


        path = os.path.join(self.raw_dir, 'node2graphID.csv')
        nodes = pd.read_csv(path, sep=',', header=0)
        nodes.index += 1

        path = os.path.join(self.raw_dir, 'graphid2label.csv')
        graphID = pd.read_csv(path, sep=',', header=0)
        graphID.index += 1


        data_list = []
        ids_list = nodes['graph_id'].unique()
        for graph_no in tqdm(ids_list):
            node_id = nodes.loc[nodes['graph_id']==graph_no, 'node_id']
            print(node_id)
            attributes = node_attrs.loc[node_id + 1, :]

            edges = edge_index.loc[edge_index['source_node'].isin(node_id)]
            edges_ids = edges.index
            label = graphID.loc[graph_no, 'label']
            print(edges)



            edges_ids = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(torch.unique(edges_ids))}
            map_edge = torch.zeros_like(edges_ids)
            for k,v in map_dict.items():
                map_edge[edges_ids==k] = v
            map_dict, map_edge, map_edge.shape
            edges_ids = map_edge.long()
            print("EI")
            print(edges_ids)
            print("EI")
            label = torch.tensor(label, dtype=torch.long)

            attrs = torch.tensor(attributes.to_numpy(),dtype=torch.float)

            graph = Data(x=attrs, edge_index=edges_ids, y=label)

            data_list.append(graph)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = NEW(root='')
dataset = dataset.shuffle()

labels = dataset.data.y
train_ratio = 0.8
train_indices = []
test_indices = []



for label in set(labels.numpy()):
    label_indices = (labels == label).nonzero(as_tuple=False).view(-1)
    train_size = int(len(label_indices) * train_ratio)
    test_size = len(label_indices) - train_size
    chosen_indices = np.random.permutation(label_indices)
    train_indices.extend(chosen_indices[:train_size].tolist())
    test_indices.extend(chosen_indices[train_size:].tolist())

train_dataset = dataset.index_select(torch.tensor(train_indices))
test_dataset = dataset.index_select(torch.tensor(test_indices))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

sampler = ImbalancedSampler(train_dataset)


print(f'Number of training classes: {train_dataset.num_classes}')
print(f'Number of test classes : {test_dataset.num_classes}')

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.5)
        self.bn1 = BatchNorm1d(128)

        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.5)
        self.bn2 = BatchNorm1d(128)

        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.5)
        self.bn3 = BatchNorm1d(128)

        self.conv4 = GraphConv(128, 128)
        self.pool4 = TopKPooling(128, ratio=0.5)
        self.bn4 = BatchNorm1d(128)

        self.conv5 = GraphConv(128, 128)
        self.pool5 = TopKPooling(128, ratio=0.5)
        self.bn5 = BatchNorm1d(128)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64 )
        self.lin3 = torch.nn.Linear(64, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x, edge_index, _, batch, _, _ = self.pool5(x, edge_index, None, batch)
        x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1+x2+x3+x4+x5
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN()



criterion = torch.nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.90)

hist = {
"epoch": [],
"train_loss": [],
"test_loss": [],
"learning_rate": [],
}

def print_losses(hist):
    s =  "| epoch | train_loss | test_loss | learning_rate |\n"
    s += "|:------|:-----------|:----------|:--------------|\n"
    for epoch, train, test, lr in zip(hist['epoch'], hist['train_loss'], hist['test_loss'], hist['learning_rate']):
        s += f"| {epoch:6} | {train:.6f}   | {test:.6f}  | {lr:.6f}       |\n"
    
    print(s)


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0
    
    if is_training:
        print("training ep...")
        model.train()
    else:
        print("eval ep...")
        model.eval()

    for data in tqdm(dataloader, total=len(dataloader)):
        model.to(device)
        data.to(device)

        if is_training:
            optimizer.zero_grad()

        labels = data.y.to(device)
        out1 = model(data)  

        loss = criterion(out1, labels)  

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / dataloader.batch_size)
        lr = scheduler.get_last_lr()[0]

    return epoch_loss / len(dataloader), lr




def train():
    for epoch in range(EPOCHS):
        print_losses(hist)
        print('Epoch[{}/{}]'.format(epoch + 1, EPOCHS))
        loss_train, lr_train = run_epoch(train_loader, is_training=True)
        loss_val, _ = run_epoch(test_loader)

        hist["epoch"].append(epoch + 1)
        hist["train_loss"].append(loss_train)
        hist["test_loss"].append(loss_val)
        hist["learning_rate"].append(lr_train)
        scheduler.step()
        print_losses(hist)


def test():
    y_pred = []
    y_true = []

    # iterate over test data
    for data in test_loader:

        model.to(device)
        data.to(device)

        model.eval()
        
        output = model(data)  # Feed Network
        
        output = torch.max(torch.exp(output), 1)[1].cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        label = data.y.cpu().numpy()
        y_true.extend(label)

    # constant for classes
    classes = ('Gmail', 'MySQL', 'Outlook', "Skype", "SMB", "BitTorrent", "Weibo", "WorldOfWarcraft")

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / cf_matrix.astype(float).sum(axis=1), index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 12), dpi=100)
    plt.title('Confusion matrix for VPN classes')
    sns.set(font_scale=2.5)
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 28}, fmt='.3f')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45)
    plt.ylabel('True')
    plt.yticks(rotation=45)
    plt.savefig('data.png')
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

train()
test()
