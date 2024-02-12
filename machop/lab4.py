import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

# figure out the correct path
machop_path = Path(".").resolve() /"machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

from torch import nn
from chop.passes.graph.utils import get_parent_name

# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear  2
            nn.Linear(16, 16),  # linear  3
            nn.Linear(16, 5),   # linear  4
            nn.ReLU(5),  # 5
        )

    def forward(self, x):
        return self.seq_blocks(x)


model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}



pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_3": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

# this performs the architecture transformation based on the config
mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})

# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)
    

# Question 1:

def create_model_and_graph():
    model = JSC_Three_Linear_Layers()
    return MaseGraph(model)

def print_graph(graph, title="Graph"):
    print(f"{title}:")
    for block, module in graph.model.seq_blocks._modules.items():
        print(f"Block number {block}: {module}")

def configure_and_apply_transformations(graph):
    pass_config = {
        "by": "name",
        "default": {"config": {"name": None}},
        "seq_blocks_2": {"config": {"name": "output_only", "channel_multiplier": 2}},
        "seq_blocks_4": {"config": {"name": "both", "channel_multiplier": 2}},
        "seq_blocks_6": {"config": {"name": "input_only", "channel_multiplier": 2}},
    }
    return redefine_linear_transform_pass(graph=graph, pass_args={"config": pass_config})

# Main execution
mg = create_model_and_graph()
print_graph(mg, "Original Graph")
mg, _ = configure_and_apply_transformations(mg)
print_graph(mg, "\nTransformed Graph")



# Question 2:
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.nn.functional import  cross_entropy

def configure_and_apply_transformations_gridsearch(graph, channel_multiplier):
    pass_config = {
        "by": "name",
        "default": {"config": {"name": None}},
        "seq_blocks_2": {"config": {"name": "output_only", "channel_multiplier": channel_multiplier}},
        "seq_blocks_4": {"config": {"name": "both", "channel_multiplier": channel_multiplier}},
        "seq_blocks_6": {"config": {"name": "input_only", "channel_multiplier": channel_multiplier}},
    }
    return redefine_linear_transform_pass(graph=graph, pass_args={"config": pass_config})

metric = MulticlassAccuracy(num_classes=5)
precision = MulticlassPrecision(num_classes=5)
recall = MulticlassRecall(num_classes=5)
f1 = MulticlassF1Score(num_classes=5)

num_batches = 5

# channel multipliers for the grid search
channel_multipliers = [1, 2, 3, 4]

# Placeholder for storing results
results = []

# Function to evaluate the model
def evaluate_model(graph, data_module, metric, precision, recall, f1, num_batches):
    recorded_metrics = []
    accs, precs, recs, f1s, losses = [], [], [], [], []
    j = 0
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = graph.model(xs)
        loss = cross_entropy(preds, ys)
        accs.append(metric(preds, ys).item())
        precs.append(precision(preds, ys).item())
        recs.append(recall(preds, ys).item())
        f1s.append(f1(preds, ys).item())
        losses.append(loss.item())
        
        if j >= num_batches - 1:
            break
        j += 1
    
    metrics_avg = {
        'accuracy': sum(accs) / len(accs),
        'precision': sum(precs) / len(precs),
        'recall': sum(recs) / len(recs),
        'f1_score': sum(f1s) / len(f1s),
        'loss': sum(losses) / len(losses)
    }
    recorded_metrics.append(metrics_avg)
    
    # Reset metrics after each evaluation
    metric.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    
    return recorded_metrics

# Main execution loop for grid search
for multiplier in channel_multipliers:
    # Reset
    mg = create_model_and_graph()
    print_graph(mg, f"Original Graph with Multiplier {multiplier}")

    # Apply transformations with the current multiplier
    mg, _ = configure_and_apply_transformations_gridsearch(mg, multiplier)
    print_graph(mg, f"\nTransformed Graph with Multiplier {multiplier}")

    # Evaluate the model
    results_for_multiplier = evaluate_model(mg, data_module, metric, precision, recall, f1, num_batches)
    
    # Store results with the multiplier for comparison
    results.append({
        'multiplier': multiplier,
        'metrics': results_for_multiplier
    })

# This comparison can be based on whichever metric
best_result = max(results, key=lambda x: x['metrics'][0]['accuracy'])  # Example based on accuracy
print(f"Best Channel Multiplier: {best_result['multiplier']} with Accuracy: {best_result['metrics'][0]['accuracy']}")



# Question 3:

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torch.nn.functional import cross_entropy


def redefine_linear_transform_pass2(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError("default value must be provided.")
    
    for node in graph.fx_graph.nodes:
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias

            # Adjusting for separate input and output channel multipliers
            channel_multiplier_input = config.get("channel_multiplier_input", 1)
            channel_multiplier_output = config.get("channel_multiplier_output", 1)
            
            if name == "output_only":
                out_features = int(out_features * channel_multiplier_output)
            elif name == "both":
                in_features = int(in_features * channel_multiplier_input)
                out_features = int(out_features * channel_multiplier_output)
            elif name == "input_only":
                in_features = int(in_features * channel_multiplier_input)
            
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    
    return graph, {}


def configure_and_apply_transformations_gridsearch(graph, channel_multiplier1, channel_multiplier2):
    pass_config = {
        "by": "name",
        "default": {"config": {"name": None}},
        "seq_blocks_2": {"config": {"name": "output_only", "channel_multiplier_output": channel_multiplier1}},
        "seq_blocks_4": {"config": {"name": "both", "channel_multiplier_input": channel_multiplier1, "channel_multiplier_output": channel_multiplier2}},
        "seq_blocks_6": {"config": {"name": "input_only", "channel_multiplier_input": channel_multiplier2}},
    }
    return redefine_linear_transform_pass2(graph=graph, pass_args={"config": pass_config})


metric = MulticlassAccuracy(num_classes=5)
precision = MulticlassPrecision(num_classes=5)
recall = MulticlassRecall(num_classes=5)
f1 = MulticlassF1Score(num_classes=5)

num_batches = 5

# Placeholder for storing results
results = []

# Function to evaluate the model
def evaluate_model(graph, data_module, metric, precision, recall, f1, num_batches):
    recorded_metrics = []
    accs, precs, recs, f1s, losses = [], [], [], [], []
    j = 0
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = graph.model(xs)
        loss = cross_entropy(preds, ys)
        accs.append(metric(preds, ys).item())
        precs.append(precision(preds, ys).item())
        recs.append(recall(preds, ys).item())
        f1s.append(f1(preds, ys).item())
        losses.append(loss.item())
        
        if j >= num_batches - 1:
            break
        j += 1
    
    metrics_avg = {
        'accuracy': sum(accs) / len(accs),
        'precision': sum(precs) / len(precs),
        'recall': sum(recs) / len(recs),
        'f1_score': sum(f1s) / len(f1s),
        'loss': sum(losses) / len(losses)
    }
    recorded_metrics.append(metrics_avg)
    
    # Reset metrics after each evaluation
    metric.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    
    return recorded_metrics


channel_multiplier_combinations = [(1, 2), (2, 3), (3, 4), (4, 1)]

for cm1, cm2 in channel_multiplier_combinations:
    mg = create_model_and_graph()
    print_graph(mg, f"Original Graph with Multipliers {cm1}, {cm2}")

    # Apply transformations with the current set of multipliers
    mg, _ = configure_and_apply_transformations_gridsearch(mg, cm1, cm2)
    print_graph(mg, f"\nTransformed Graph with Multipliers {cm1}, {cm2}")

    # Evaluate the model
    results_for_combination = evaluate_model(mg, data_module, metric, precision, recall, f1, num_batches)
    
    # Store results with the multiplier combination for comparison
    results.append({
        'multipliers': (cm1, cm2),
        'metrics': results_for_combination
    })

best_result = max(results, key=lambda x: x['metrics'][0]['accuracy'])
print(f"Best Channel Multipliers: {best_result['multipliers']} with Accuracy: {best_result['metrics'][0]['accuracy']}")
