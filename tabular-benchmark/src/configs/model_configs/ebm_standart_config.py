import numpy as np

config_random = {"model_type": {
    "value": "sklearn"
},
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__max_bins": {
        "values": [128, 256, 512]
    },
    "model__learning_rate": {
        'distribution': "log_uniform_values",
        'min': 1E-4, 
        'max': 0.7,
    },    
    "model__validation_size": {
            "value": 0.20
    },
    "model__interactions": {
        "values": [0]
    },
    "model__max_rounds": {
        "value": 20000
    },
    "model__min_samples_leaf": {
        "distribution": "int_uniform",
        'min': 1,
        'max': 100
    },
    "model__inner_bags": {
        "values":[0, 5, 10, 15]
    },
     "model__outer_bags": {
         "values":[8, 16, 32, 64, 128]
     },   
    "transformed_target": {
        "values": [False,True]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
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
        "value": True
    },
    "use_gpu": {
        "value": False
    }
}

config_regression = dict(config_random, **{
    "model_name": {
        "value": "ebm_standart_r"
    },
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "ebm_standart_r"
    },
})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "ebm_standart_c"
    },
})

config_classif_default= dict(config_default, **{
    "model_name": {
        "value": "ebm_standart_c"
    },
})
