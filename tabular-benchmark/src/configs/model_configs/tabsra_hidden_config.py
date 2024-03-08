from configs.model_configs.skorch_config import skorch_config, skorch_config_default

config_random = {
    "model__module__n_head": {
        "distribution": "q_uniform",
        "min": 1,
        "max": 2
    },
    
    "model__module__n_hidden_encoder": {
        'values': [1]
    },
    "model__module__dim_head": {
        'values': [4, 8, 12]
    },
    "model__module__dropout_rate": {
        "distribution": "uniform",
        "min": 0,
        "max": 0.5
    },
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
    "model__module__encoder_bias": {
        "value": True
    }, 
    "model__module__classifier_bias": {
        "value": True
    },
    "model__module__get_attention": {
        "value": False
    },
}

config_default = {
    "model__module__n_head": {
        "value": 1
    },
    "model__module__dim_head": {
        "value": 8
    },
    "model__module__dropout_rate": {
        "value": 0.2
    },
    "model__lr": {
        "value": 0.001
    },
    "model__optimizer__weight_decay": {
        "value": 1e-5
    },
    "model__lr_scheduler": {
        "value": False
    },
    "model__module__encoder_bias": {
        "value": True
    },
    "model__module__classifier_bias": {
        "value": True
    },
    "model__module__get_attention": {
        "value": False
    },
    "use_gpu": {
        "value": False
    },
    
}

config_regression = dict(config_random,
                                        **skorch_config,
                                        **{
                                            "model_name": {
                                                "value": "tabsra_hidden_regressor"
                                            },
                                        })

config_regression_default = dict(config_default,
                                        **skorch_config_default,
                                        **{
                                            "model_name": {
                                                "value": "tabsra_hidden_regressor"
                                            },
                                        })

config_classif = dict(config_random,
                                     **skorch_config,
                                     **{
                                         "model_name": {
                                             "value": "tabsra_hidden"
                                         },
                                     })

config_classif_default = dict(config_default,
                                     **skorch_config_default,
                                     **{
                                         "model_name": {
                                             "value": "tabsra_hidden"
                                         },
                                     })
