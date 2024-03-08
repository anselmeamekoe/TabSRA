import numpy as np

config_random = {
    "model_type": {
        "value": "sklearn"
    },
    "model__max_depth": {
        "values": [2, 3, 4, 5, 6, 7],
    },
    "model__min_samples_split": {
        "values": [2, 3],
    },
    "model__min_samples_leaf": {  # Not exactly like Hyperopt
        "distribution": "q_log_uniform_values",
        "min": 1.5,
        "max": 50.5,
        "q": 1
    },
    "model__max_features": {
        "values": ["auto", "sqrt", "log2"]
    },
    "transformed_target": {
        "values": [False, True]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
}

config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
    "use_gpu": {
        "value": False
    }
}

config_regression = dict(config_random, **{
    "model_name": {
        "value": "dt_r"
    },
    "model__criterion": {
        "values": ["squared_error", "absolute_error"],
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "dt_r"
    },

})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "dt_c"
    },
    "model__criterion": {
        "values": ["gini", "entropy"],
    },
})

config_classif_default = dict(config_default, **{
    "model_name": {
        "value": "dt_c"
    },
})
