from configs.model_configs.skorch_config import skorch_config, skorch_config_default

config_random = {
    "model__lr": {
        "distribution": "log_uniform_values",
        "min": 1e-5,
        "max": 1e-2
    },
    "model__optimizer__weight_decay": {
        "distribution": "log_uniform_values",
        "min": 1e-6,
        "max": 1e-4
    },
    "model__lr_scheduler": {
        "values": [True, False]
    },
    "use_gpu": {
        "value": False
    }, 
    "model__module__bias": {
        "value": True
    }
}

config_default = {
    "model__lr": {
        "value": 0.001
    },
    "model__optimizer__weight_decay": {
        "value": 1e-5
    },
    "model__lr_scheduler": {
        "value": False
    },
    "model__module__bias": {
        "value": True
    },
    "use_gpu": {
        "value": False
    },
    
}

config_regression = dict(config_random,
                                        **skorch_config,
                                        **{
                                            "model_name": {
                                                "value": "linear_regressor"
                                            },
                                        })

config_regression_default = dict(config_default,
                                        **skorch_config_default,
                                        **{
                                            "model_name": {
                                                "value": "linear_regressor"
                                            },
                                        })

config_classif = dict(config_random,
                                     **skorch_config,
                                     **{
                                         "model_name": {
                                             "value": "linear"
                                         },
                                     })

config_classif_default = dict(config_default,
                                     **skorch_config_default,
                                     **{
                                         "model_name": {
                                             "value": "linear"
                                         },
                                     })
