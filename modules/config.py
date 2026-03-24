import importlib
import inspect
import json
from typing import Any, Dict, List, Union

def load_model(config_path: str) -> Any:
    
    with open(config_path, "r") as f:
        config = json.load(f)

    # 动态导入类
    class_path = config["model_type"].split(".")
    module_name = ".".join(class_path[:-1])
    class_name = class_path[-1]
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)

    # 获取构造函数签名
    init_params = inspect.signature(model_class.__init__).parameters
    valid_params = {}

    # 自动处理参数
    for param_name, param_spec in init_params.items():
        if param_name == "self":
            continue

        # 从配置中获取参数值（优先使用配置，否则用默认值）
        param_value = config["params"].get(param_name, param_spec.default)

        # 处理默认值缺失的情况
        if param_value is inspect.Parameter.empty:
            raise ValueError(f"Required parameter '{param_name}' missing in config")

        # 自动将列表转换为元组（针对特定参数）
        if param_name in {"num_res_blocks", "num_channels", "attention_levels", "num_head_channels"}:
            if isinstance(param_value, list):
                param_value = tuple(param_value)

        valid_params[param_name] = param_value

    return model_class(**valid_params)
