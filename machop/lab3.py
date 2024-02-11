import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp
from logging import getLogger

# figure out the correct path
machop_path = Path(".").resolve().parent /"mase/machop"
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger

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


logger = getLogger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)


pass_args = {
"by": "type",
"default": {"config": {"name": None}},
"linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
},}

import copy
# build a search space
data_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
w_in_frac_widths = [(16, 8), (8, 6), (8, 4), (4, 2)]
search_spaces = []
for d_config in data_in_frac_widths:
    for w_config in w_in_frac_widths:
        pass_args['linear']['config']['data_in_width'] = d_config[0]
        pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
        pass_args['linear']['config']['weight_width'] = w_config[0]
        pass_args['linear']['config']['weight_frac_width'] = w_config[1]
        # dict.copy() and dict(dict) only perform shallow copies
        # in fact, only primitive data types in python are doing implicit copy when a = b happens
        search_spaces.append(copy.deepcopy(pass_args))



# grid search

import torch
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

# Initialize metrics
metric = MulticlassAccuracy(num_classes=5)
precision = MulticlassPrecision(num_classes=5)
recall = MulticlassRecall(num_classes=5)
f1 = MulticlassF1Score(num_classes=5)

num_batches = 5

# Initialize records
recorded_metrics = []

for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0

    # Initialize accumulators for metrics
    accs, precs, recs, f1s, losses = [], [], [], [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mg.model(xs)
        
        # Calculate metrics
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        prec = precision(preds, ys)
        rec = recall(preds, ys)
        f1_score = f1(preds, ys)
        
        # Append calculated metrics
        accs.append(acc.item())
        precs.append(prec.item())
        recs.append(rec.item())
        f1s.append(f1_score.item())
        losses.append(loss.item())
        
        if j >= num_batches:
            break
        j += 1

    # Calculate and record averages
    metrics_avg = {
        'accuracy': sum(accs) / len(accs),
        'precision': sum(precs) / len(precs),
        'recall': sum(recs) / len(recs),
        'f1_score': sum(f1s) / len(f1s),
        'loss': sum(losses) / len(losses)
    }
    recorded_metrics.append(metrics_avg)

    metric.reset()
    precision.reset()
    recall.reset()
    f1.reset()

print(recorded_metrics)