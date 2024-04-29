import numpy as np

config_random = {"model_type": {
    "value": "sklearn"
},
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_depth": {
        "values": [3, 4, 5, 6, 7, 8, 9, 10, 11]
    },
    "model__learning_rate": {
        'distribution': "log_uniform_values",
        'min': 1E-5,  # inspired by RTDL
        'max': 0.7,
    },
    "model__n_estimators": {
        "value": 1_000,
        #"distribution": "q_uniform",
        #"min": 100,
        #"max": 6000,
        #"q": 200
    },
    "model__l2_leaf_reg": {
        "distribution": "log_uniform_values",
        'min': 1,
        'max': 10
    },    
    "early_stopping_rounds": {
        "value": 20
    },
    "transformed_target": {
        "values": [False, True]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": False
    },
    "use_gpu": {
        "value": False
    }
}


config_default = {
    "model_type": {
        "value": "sklearn"
    },
    "transformed_target": {
        "values": [False]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": False
    },
    "use_gpu": {
        "value": False
    }
}

config_regression = dict(config_random, **{
    "model_name": {
        "value": "catboost_r"
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "catboost_r"
    },
})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "catboost_c"
    },
})

config_classif_default= dict(config_default, **{
    "model_name": {
        "value": "catboost_c"
    },
})
