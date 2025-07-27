import json
import os
from copy import deepcopy
"""
    Copyright (C) 2024 LeratLy - All Rights Reserved
    You may use, distribute and modify this code under the
    terms of the MIT license.
    You should have received a copy of the MIT license with
    this file.

    Copyright (c) 2021 VISTEC - Vidyasirimedhi Institute of Science and Technology.
    Code of VISTEC has been used and modified in this file under terms of the MIT license.
"""


class ConfigFunctionalities:
    """
    Enables a class to use different functionalities which are helpful for different config setups
    """

    def clone(self):
        return deepcopy(self)

    def inherit(self, another):
        """inherit common keys from a given config"""
        common_keys = set(self.__dict__.keys()) & set(another.__dict__.keys())
        for k in common_keys:
            setattr(self, k, getattr(another, k))

    def propagate(self):
        """push down the configuration to all members"""
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigFunctionalities):
                v.inherit(self)
                v.propagate()

    def save(self, save_path):
        """save config to json file"""
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        conf = self.as_dict_jsonable()
        with open(save_path, 'w') as f:
            json.dump(conf, f)

    def load(self, load_path):
        """load json config"""
        with open(load_path) as f:
            conf = json.load(f)
        self.from_dict(conf)

    def from_dict(self, dict, strict=False):
        for k, v in dict.items():
            if not hasattr(self, k):
                if strict:
                    raise ValueError(f"loading extra '{k}'")
                else:
                    print(f"loading extra '{k}'")
                    continue
            try:
                if isinstance(self.__dict__[k], ConfigFunctionalities):
                    self.__dict__[k].from_dict(v)
                else:
                    self.__dict__[k] = v
            except KeyError:
                setattr(self, k, v)

    def as_dict_jsonable(self):
        conf = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigFunctionalities):
                conf[k] = v.as_dict_jsonable()
            else:
                if jsonable(v):
                    conf[k] = v
                # ignore not jsonable
                else:
                    pass
        return conf


def jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False
