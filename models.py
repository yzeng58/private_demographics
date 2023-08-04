import torchvision, torch
import torch.nn as nn
from transformers import BertForSequenceClassification

def get_resnet50(n_classes, pretrained, seed = 123):
    torch.manual_seed(seed)
    model = torchvision.models.resnet50(pretrained=pretrained)
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
    model = torch.nn.DataParallel(model)
    return model

def get_resnet18(n_classes, pretrained, seed = 123):
    torch.manual_seed(seed)
    model = torchvision.models.resnet18(pretrained=pretrained)
    d = model.fc.in_features
    model.fc = nn.Linear(d, n_classes)
    return model

class BertWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(
            input_ids=x[:, :, 0],
            attention_mask=x[:, :, 1],
            token_type_ids=x[:, :, 2]).logits


def get_bert_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8
    )
    return optimizer

def get_bert(n_classes):
    model = BertWrapper(
                BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased', num_labels=n_classes)
            )
    model.zero_grad()
    return model

class logReg(torch.nn.Module):
    """
    Logistic regression model.
    """
    def __init__(self, num_features, num_classes, seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x.float())
        probas = torch.sigmoid(logits)
        return probas.type(torch.FloatTensor), logits.type(torch.FloatTensor)

class mlp(torch.nn.Module):
    """
    Multilayer Perceptron
    """
    def __init__(self, num_features, num_classes, seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear1 = torch.nn.Linear(num_features, 50)
        self.linear2 = torch.nn.Linear(50, 50)
        self.linear3 = torch.nn.Linear(50, 50)
        self.linear4 = torch.nn.Linear(50, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.linear1(x.float())
        out = self.relu(out)

        out = self.linear2(out)
        out = self.relu(out)

        out = self.linear3(out)
        out = self.relu(out)

        out = self.linear4(out)
        probas = torch.sigmoid(out)

        return probas.type(torch.FloatTensor), out

class LogReg(torch.nn.Module):
    """
    Multilayer Perceptron
    """
    def __init__(self, num_features, num_classes, seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.linear(x.float())
        probas = torch.sigmoid(out)
        return probas.type(torch.FloatTensor), out

class cmnist_mlp(torch.nn.Module):
    def __init__(
        self, 
        grayscale_model = False,
        hidden_dim = 256,
        seed = 123,
    ):
        super().__init__()
        torch.manual_seed(seed)

        if grayscale_model:
            lin1 = torch.nn.Linear(14 * 14, hidden_dim)
        else:
            lin1 = torch.nn.Linear(2 * 14 * 14, hidden_dim)

        lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        lin3 = torch.nn.Linear(hidden_dim, 2)

        for lin in [lin1, lin2, lin3]:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

        self._main = torch.nn.Sequential(
            lin1, 
            torch.nn.ReLU(True), 
            lin2, 
            torch.nn.ReLU(True), 
            lin3
        )
        self.grayscale_model = grayscale_model

    def forward(self, input):
        if self.grayscale_model:
            out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        probas = torch.sigmoid(out)
        return probas.type(torch.FloatTensor), out

def load_model(
    model = 'mlp',
    num_feature = 3,
    num_class = 2,
    seed = 123,
):
    if model == 'resnet50':
        model_list = get_resnet50(num_class, True, seed)
    elif model == 'resnet18':
        model_list = get_resnet18(num_class, True, seed)
    elif model == 'mlp':
        model_list = mlp(num_feature, num_class, seed)
    elif model == 'logreg':
        model_list = LogReg(num_feature, num_class, seed)
    elif model == 'bert':
        model_list = get_bert(num_class)
    elif model == 'cmnist_mlp':
        model_list = cmnist_mlp(False, 256, seed)
    return model_list
    