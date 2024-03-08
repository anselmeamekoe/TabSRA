import numpy as np

config_random = {"model_type": {
    "value": "sklearn"
},
    # Parameter space taken from Hyperopt-sklearn except when mentioned
    "model__n_hid": {
        "values": [5, 7, 10, 15, 20]
    },
    "model__n_estimators": {
        "values": [50000]
    },
    "model__boost_rate": {
        "values": [0.1, 0.3, 0.5,  0.7, 1.0]
    }, 
    "model__init_reg": {
        "distribution": "log_uniform_values",
        'min': 1E-7,  # inspired by RTDL
        'max': 1E0,
    },
    "model__elm_alpha": {
        "distribution": "log_uniform_values",
        'min': 1E-7,  # inspired by RTDL
        'max': 1E0,
    },
    "model__elm_scale": {
        "values": [1, 3, 5,  10]
    }, 
    "transform__0__method_name": {
        "value": "gaussienize"
    },
    "transform__0__type": {
        "value": "standard",
    }, 
    "transform__0__apply_on": {
        "value": "numerical",
    },
    "transformed_target": {
        "values": [True]
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
    "transform__0__method_name": {
        "value": "gaussienize"
    },
    "transform__0__type": {
        "value": "standard",
    }, 
    "transform__0__apply_on": {
        "value": "numerical",
    },
    "transformed_target": {
        "values": [True]
    },
    "one_hot_encoder": {  # Use one-hot encoding for categorical variables when needed
        "value": True
    },
    "use_gpu": {
        "value": False
    },
}

config_regression = dict(config_random, **{
    "model_name": {
        "value": "igann_r"
    },
    "model__task": {
        "value": "regression"
    }
    
})

config_regression_default = dict(config_default, **{
    "model_name": {
        "value": "igann_r"
    },
    "model__task": {
        "value": "regression"
    }
})

config_classif = dict(config_random, **{
    "model_name": {
        "value": "igann_c"
    },
    "model__task": {
        "value": "classification"
    }
})

config_classif_default= dict(config_default, **{
    "model_name": {
        "value": "igann_c"
    },
    "model__task": {
        "value": "classification"
    }
})
