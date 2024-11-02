from dgl.nn.pytorch import GATConv
import torch
import torch.nn as nn
import torch as th
import math
class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block with one fully connected layer.

    Args:
        in_features (int): The number of input features.
        output_size (int): The number of output features.
        bias (boolean): Add bias to the linear layer
        layer_norm (boolean): Apply layer normalization
        dropout (float): The dropout value
        activation (nn.Module): The activation function to be applied after each fully connected layer.

    Example:
    ```python
    # Create an MLP block with 2 hidden layers and ReLU activation
    mlp_block = MLPBlock(input_size=64, output_size=10, activation=nn.ReLU())

    # Apply the MLP block to an input tensor
    input_tensor = torch.randn(32, 64)
    output = mlp_block(input_tensor)
    ```
    """
    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.75, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x
class Residual(nn.Module):
    """
    A residual layer that adds the output of a function to its input.

    Args:
        fn (nn.Module): The function to be applied to the input.

    """

    def __init__(self, fn):
        """
        Initialize the Residual layer with a given function.

        Args:
            fn (nn.Module): The function to be applied to the input.
        """
        super().__init__()
        self.fn = fn

    def forward(self, x):
        """
        Forward pass of the Residual layer.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: The input tensor added to the result of applying the function `fn` to it.
        """
        return x + self.fn(x)

class MLPModel(nn.Module):
    """
    Baseline MLP model with two fully connected layers with residual connection
    """

    def __init__(self, input_length, nb_gos, device, nodes=[1024, ]):
        super().__init__()
        self.nb_gos = nb_gos
        net = []
        for hidden_dim in nodes:
            net.append(MLPBlock(input_length, hidden_dim))
            net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
            input_length = hidden_dim
        net.append(nn.Linear(input_length, nb_gos))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

    def forward(self, features):
        return self.net(features)
class BaseModel(nn.Module):
    """
    A base model with ElEmbeddings loss functions
    Args:
        input_length (int): The number of input features
        nb_gos (int): The number of Gene Ontology (GO) classes to predict
        nb_zero_gos (int): The number of GO classes without training annotations
        nb_rels (int): The number of relations in GO axioms
        device (string): The compute device (cpu:0 or gpu:0)
        hidden_dim (int): The hidden dimension for an MLP
        embed_dim (int): Embedding dimension for GO classes and relations
        margin (float): The margin parameter of ELEmbedding method
    """

    def __init__(self, input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=2560, embed_dim=2560, margin=0.1):
        super().__init__()
        self.nb_gos = nb_gos
        self.nb_zero_gos = nb_zero_gos
        self.nb_rels = nb_rels
        # ELEmbedding Model Layers
        self.embed_dim = embed_dim
        # Create additional index for hasFunction relation
        self.hasFuncIndex = th.LongTensor([nb_rels])
        # Embedding layer for all1.csv classes in GO
        self.go_embed = nn.Embedding(nb_gos + nb_zero_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        # Initialize embedding layers
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos + nb_zero_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        self.rel_embed = nn.Embedding(nb_rels + 1, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        # indices of all1.csv GO classes
        self.all_gos = th.arange(self.nb_gos)
        self.margin = margin
class SharedCoreDeepGATModel(nn.Module):
    def __init__(self, shared_input_length, shared_hidden_dim, shared_embed_dim):
        super().__init__()
        # 共享的核心模型部分
        self.shared_net1 = MLPBlock(shared_input_length, shared_hidden_dim)
        self.shared_conv1 = GATConv(shared_hidden_dim, shared_embed_dim, num_heads=1)
        # 保存旧参数的字典
        self.old_params = {}
        self.fisher_information = None
    def forward_shared(self, features, g1):
        x = self.shared_net1(features)
        x = self.shared_conv1(g1, x).squeeze(dim=1)
        return x
    def save_old_parameters(self):
        """保存共享模型的旧参数。"""
        for name, param in self.named_parameters():
            self.old_params[name] = param.clone().detach()
    def ewc_loss(self, fisher_information, lambda_ewc=0.5):
        loss = 0
        if self.old_params is not None:
            for name, param in self.named_parameters():
                if name in self.old_params:
                    loss += (fisher_information[name] * (param - self.old_params[name]) ** 2).sum()
        return lambda_ewc * loss
class TaskSpecificModel(BaseModel):
    def __init__(self, shared_model, input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim=2560, embed_dim=2560):
        super().__init__(input_length, nb_gos, nb_zero_gos, nb_rels, device, hidden_dim, embed_dim)
        self.shared_model = shared_model
        self.task_net = nn.Sequential(
            nn.Linear(embed_dim, nb_gos),
            nn.Sigmoid())
    def forward(self,input_nodes, output_nodes, blocks):
        g1 = blocks[0]
        features = g1.ndata['feat']['_N']
        shared_features = self.shared_model.forward_shared(features, g1)
        logits = self.task_net(shared_features)
        return logits
