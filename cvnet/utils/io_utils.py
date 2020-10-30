# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
import os


def load_json(json_path, encoding='utf-8'):
    """
    Loads a json config from a file.
    """
    assert os.path.exists(json_path), "Json file %s not found" % json_path
    with open(json_path, 'r', encoding=encoding) as f:
        json_config = f.read()
    try:
        config = json.loads(json_config)
    except BaseException as err:
        raise Exception("Failed to validate config with error: %s" % str(err))

    return config


def save_json(data, json_path, mode='w', encoding='utf-8'):
    dir = os.path.dirname(os.path.abspath(json_path))
    if not os.path.exists(dir):
        print(dir)
        os.makedirs(dir)
    with open(json_path, mode=mode, encoding=encoding) as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    p = '../configs/resnet50_imagenet_classy_config.json'
    a = load_json(p)
    print(a)
    with open(p) as f:
        b = json.load(f)
        print(b)