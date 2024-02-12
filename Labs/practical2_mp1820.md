# Lab 3

### 1) Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
In addtion to the quality metrics mentioned in the question, we can implement for example F1 score, precision and recall as metrics. The meaning of the metrics are explained below.


Precision:

$ \text{Precision} = \frac{TP}{TP + FP} $

- **TP (True Positives):** Correct positive predictions.
- **FP (False Positives):** Incorrect positive predictions.

Precision assesses the quality of positive class predictions, indicating the proportion of positive identifications that were actually correct.



Recall:
Recall measures the model's ability to detect all relevant instances.


$ Recall = \frac{TP}{TP + FN}$

- **FN (False Negatives):** Positive cases incorrectly predicted as negative.

Recall indicates the ability of the model to find all the actual positives, showing how many positives the model can retrieve from the total positive instances.



F1 Score:

The F1 Score is the harmonic mean of precision and recall, balancing the two metrics.

$ \text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $

The F1 Score incorporates both false positives and false negatives, providing a single metric that reflects the model's precision and recall balance. A higher F1 Score indicates better model performance in terms of precision and recall balance.


### 2) Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).
```python 
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
```


### 3) Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.
The brute-force search was implemented by adding a case statement to optuna.py:
```python
case "brute-force":
    sampler = optuna.samplers.BruteForceSampler()
```
And by changing the sampling parameter in [search.strategy.setup] to "brute_force" .


### 4) Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.
By running tpe search, we get:

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:09<00:00,  2.13it/s, 9.39/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        0 | {'loss': 1.273, 'accuracy': 0.528} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.528, 'average_bitwidth': 0.8} |
|  1 |       16 | {'loss': 1.291, 'accuracy': 0.519} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.519, 'average_bitwidth': 0.4} |
INFO     Searching is completed



By running brute-force search, we get:

90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                    | 18/20 [00:11<00:01,  1.58it/s, 11.40/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        3 | {'loss': 1.309, 'accuracy': 0.524} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.524, 'average_bitwidth': 1.6} |
|  1 |        4 | {'loss': 1.309, 'accuracy': 0.521} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.521, 'average_bitwidth': 0.8} |
|  2 |        8 | {'loss': 1.301, 'accuracy': 0.521} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.521, 'average_bitwidth': 0.4} |
|  0 |        0 | {'loss': 1.273, 'accuracy': 0.528} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.528, 'average_bitwidth': 0.8} |
|  1 |        1 | {'loss': 1.28, 'accuracy': 0.527}  | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.527, 'average_bitwidth': 0.4} |
|  2 |        3 | {'loss': 1.279, 'accuracy': 0.528} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.528, 'average_bitwidth': 1.6} |
INFO     Searching is completed

From the executions, we clearly see 3 main points:

- the brute-force search took 11.4 seconds to complete which is comparable to the 9.8s of the tpe search. 
- the best accuracy and losses attained by the two search methods are almost identical, with tpe just getting the edge with an accuracy of 0.519 compared to brute-force's 0.521. 
- tpe search could only find 2 optimal quantizations strategies considerably worse compared to 6 found by the brute-force method.


# Lab 4

### 1) Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.
In order to make the layers expanded to double the size we can execute the following:
```python
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
```



### 2) In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

Following the same strcuture as lab3, we can adapt the grid search to now search the best channel multiplier value.

```python
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
```

Results:
Best Channel Multiplier: 4 with Accuracy: 0.24000000357627868



### 3) You may have noticed, one problem with the channel multiplier is that it scales all layers uniformly, ideally, we would like to be able to construct networks like the following: 
```python
# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.ReLU(16),
            nn.Linear(16, 32),  # output scaled by 2
            nn.ReLU(32),  # scaled by 2
            nn.Linear(32, 64),  # input scaled by 2 but output scaled by 4
            nn.ReLU(64),  # scaled by 4
            nn.Linear(64, 5),  # scaled by 4
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)
```
Can you then design a search so that it can reach a network that can have this kind of structure?

Yes, by defining a new redefine_linear_transform_pass that accepts various multipliers:
```python
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
```

And running the grid search loop:

```python
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
```

We get the output:
Best Channel Multipliers: (3, 4) with Accuracy: 0.19000000059604644