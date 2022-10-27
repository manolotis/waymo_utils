import numpy as np


def validate_agent_type(agent_type):
    acceptable_types = [str, int, float]
    for t in acceptable_types:
        if type(agent_type) is t:
            return agent_type
    if type(agent_type) is np.ndarray:
        agent_type = agent_type.tolist()
    if type(agent_type) is list and len(agent_type) == 1:
        return agent_type[0]

    raise TypeError(f"Agent type was not one of: str, int or float. Was: {type(agent_type)}")


def type_str2int(agent_type):
    str2int = {
        "vehicle": 1,
        "pedestrian": 2,
        "cyclist": 3
    }
    return str2int[agent_type]


def type_int2str(agent_type):
    int2str = {
        1: "vehicle",
        2: "pedestrian",
        3: "cyclist"
    }
    return int2str[agent_type]


def type_to_int(agent_type):
    agent_type = validate_agent_type(agent_type)
    t = type(agent_type)
    if t is str:
        return type_str2int(agent_type)
    if t is int:
        return agent_type
    if t is float:
        return int(agent_type)


def type_to_str(agent_type):
    agent_type = validate_agent_type(agent_type)
    t = type(agent_type)
    if t is str:
        return agent_type
    if t is int:
        return type_int2str(agent_type)
    if t is float:
        return type_int2str(int(agent_type))
    else:
        print(agent_type)
        raise TypeError(f"Type: {type(agent_type)}")


def print_data_keys(data, end_execution=True):
    for k in data.keys():
        try:
            print(k, data[k].shape)
        except:
            print("Key did not have .shape: ", k, type(data[k]))

    if end_execution:
        exit()
