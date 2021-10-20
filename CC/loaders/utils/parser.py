from typing import Dict
from distutils.util import strtobool
import json


class KwargItem():
    def __init__(self, name: str, _type: type, optional: bool = False, defaultValue=None, description: str = ""):
        self.name = name
        self.type = _type
        if defaultValue is not None:
            optional = True
        self.optional = optional
        self.defaultValue = defaultValue
        if description == "":
            if optional:
                desription = f"args {name}: required!"
        self.desription = description


class KwargsParser():
    def __init__(self, debug: bool = False):
        self.args_name_dict: Dict[KwargItem] = {}
        self.debug = debug
        self.kv = {}

    def add_argument(self, argName: str, argType: type, defaultValue=None, optional: bool = False, description: str = ""):
        """add argument

        Args:
            argName (str): arg name
            argType (type): arg type
            defaultValue ([type], optional): [description]. Defaults to None.
            optional (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        self.args_name_dict[argName] = KwargItem(
            argName, argType, optional=optional, defaultValue=defaultValue, description=description)
        return self

    def parse(self, instance, **kwargs):
        self.kv = {}
        for argName in self.args_name_dict:
            arg: KwargItem = self.args_name_dict[argName]
            value = arg.defaultValue
            if not arg.optional:
                if argName not in kwargs:
                    raise ValueError(arg.description)
                value = kwargs[argName]
            else:
                if argName in kwargs:
                    value = kwargs[argName]
            setattr(instance, argName, self._convert_to(value, arg.type))
            self.kv[argName] = getattr(instance, argName)
        if self.debug:
            print(f"kwargs parser: {json.dumps(self.kv,indent=4)}")
        return self

    def parse_dict(self, **kwargs):
        self.kv = {}
        for argName in self.args_name_dict:
            arg: KwargItem = self.args_name_dict[argName]
            value = arg.defaultValue
            if not arg.optional:
                if argName not in kwargs:
                    raise ValueError(arg.description)
                value = kwargs[argName]
            else:
                if argName in kwargs:
                    value = kwargs[argName]
            self.kv[argName] = self._convert_to(value, arg.type)
        if self.debug:
            print(f"kwargs parser: {json.dumps(self.kv,indent=4)}")
        return self.kv

    def _convert_to(self, value, _type: type):
        if isinstance(value, str):
            if _type is bool:
                return bool(strtobool(value))
            else:
                return _type(value)
        elif isinstance(value, _type):
            return value
        else:
            raise ValueError(f"value {value} could not convert to {_type}")
