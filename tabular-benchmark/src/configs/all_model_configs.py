import sys
sys.path.append("src")
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from  igann import IGANN
from catboost import CatBoostClassifier, CatBoostRegressor
from models.skorch_models import create_resnet_skorch, create_ft_transformer_skorch, create_rtdl_mlp_skorch, create_tabsra_skorch, create_linear_skorch
from models.skorch_models_regression import create_resnet_regressor_skorch, create_ft_transformer_regressor_skorch, create_rtdl_mlp_regressor_skorch, \
    create_tabsra_regressor_skorch, create_linear_regressor_skorch
from models.TabSurvey.models.saint import SAINT




total_config = {}
model_keyword_dic = {}

#### ADD CatBoost ### 
from configs.model_configs.catboost_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "catboost"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = CatBoostRegressor 
model_keyword_dic[config_classif["model_name"]["value"]] = CatBoostClassifier

#### ADD IGANN ### 
from configs.model_configs.igann_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "igann"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = IGANN
model_keyword_dic[config_classif["model_name"]["value"]] = IGANN


#### ADD DT ### 
from configs.model_configs.dt_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "dt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = DecisionTreeRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = DecisionTreeClassifier

#### ADD EBM Standart ### 
from configs.model_configs.ebm_standart_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "ebm_standart"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = ExplainableBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = ExplainableBoostingClassifier

#### ADD EBM ### 
from configs.model_configs.ebm_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "ebm"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = ExplainableBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = ExplainableBoostingClassifier

## ADD Linear ##
from configs.model_configs.linear_config import config_classif, config_regression, config_classif_default, config_regression_default
#replace template.py by your parameters
keyword = "linear"
total_config[keyword] = {
         "classif": {"random": config_classif,
                     "default": config_classif_default},
         "regression": {"random": config_regression,
                             "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = create_linear_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_linear_skorch

## ADD TabSRA_hidden ##
from configs.model_configs.tabsra_hidden_config import config_classif, config_regression, config_classif_default, config_regression_default
#replace template.py by your parameters
keyword = "tabsra_hidden"
total_config[keyword] = {
         "classif": {"random": config_classif,
                     "default": config_classif_default},
         "regression": {"random": config_regression,
                             "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = create_tabsra_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_tabsra_skorch

## ADD TabSRA_hidden_h

## ADD TabSRA_h ##
from configs.model_configs.tabsra_hidden_h_config import config_classif, config_regression, config_classif_default, config_regression_default
#replace template.py by your parameters
keyword = "tabsra_hidden_h"
total_config[keyword] = {
         "classif": {"random": config_classif,
                     "default": config_classif_default},
         "regression": {"random": config_regression,
                             "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = create_tabsra_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_tabsra_skorch

## ADD TabSRA_h ##
from configs.model_configs.tabsra_h_config import config_classif, config_regression, config_classif_default, config_regression_default
#replace template.py by your parameters
keyword = "tabsra_h"
total_config[keyword] = {
         "classif": {"random": config_classif,
                     "default": config_classif_default},
         "regression": {"random": config_regression,
                             "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = create_tabsra_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_tabsra_skorch

## ADD TabSRA ##
from configs.model_configs.tabsra_config import config_classif, config_regression, config_classif_default, config_regression_default
#replace template.py by your parameters
keyword = "tabsra"
total_config[keyword] = {
         "classif": {"random": config_classif,
                     "default": config_classif_default},
         "regression": {"random": config_regression,
                             "default": config_regression_default},
}
#these constructor should create an object
# with fit and predict methods
model_keyword_dic[config_regression["model_name"]["value"]] = create_tabsra_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_tabsra_skorch
#############################


from configs.model_configs.gpt_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "gbt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = GradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = GradientBoostingClassifier


from configs.model_configs.rf_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "rf"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = RandomForestRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = RandomForestClassifier

from configs.model_configs.hgbt_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "hgbt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = HistGradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = HistGradientBoostingClassifier

from configs.model_configs.xgb_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = XGBClassifier

from configs.model_configs.xgb_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = XGBClassifier

from configs.model_configs.mlp_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "mlp"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_rtdl_mlp_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_rtdl_mlp_skorch

from configs.model_configs.resnet_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "resnet"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_resnet_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_resnet_skorch

from configs.model_configs.ft_transformer_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "ft_transformer"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_ft_transformer_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_ft_transformer_skorch

from configs.model_configs.saint_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "saint"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = SAINT
model_keyword_dic[config_classif["model_name"]["value"]] = SAINT


