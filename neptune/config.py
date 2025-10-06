import os 
import json
import importlib 
import inspect
import sys 

class ConfigurationError(Exception):
    pass

class ConflictError(Exception):
    pass

# Use snake-case (.e.g. "my_project") for the project name
PROJECT_NAME = "neptune"
PROJECT_VERSION = "0.1.0"
PROJECT_ENVVAR_PREFIX = PROJECT_NAME.upper()
CONFIG_FILE_NAME = PROJECT_NAME + ".json"
DEFAULT_PLUGINS = [
    "neptune.providers.*"
]

def get_config_location():
    return os.environ.get(PROJECT_ENVVAR_PREFIX + "_CONFIG", os.path.join(os.path.expanduser("~"), ".config", CONFIG_FILE_NAME))

def get_config(key: str, default, doc, expected_type=str, obj = None):
    if obj is not None and hasattr(obj, key):
        return expected_type(getattr(obj, key))
    config = {} 
    envvar = PROJECT_ENVVAR_PREFIX + "_" + key.upper()
    if envvar in os.environ:
        if expected_type == bool:
            return os.environ[envvar].lower() in ["true", "1", "yes", "y", "YES", "Y", "True", "TRUE"]
        if expected_type == list:
            return os.environ[envvar].split(" ")
        return expected_type(os.environ[envvar])
    default_config_location = os.path.join(os.path.expanduser("~"), ".config", CONFIG_FILE_NAME)
    config_location = os.environ.get(PROJECT_ENVVAR_PREFIX +  "_CONFIG", default_config_location)
    if not os.path.exists(config_location) and config_location != default_config_location:
        raise ConfigurationError(f"Configuration file {config_location} does not exist")
    if not os.path.exists(config_location):
        return expected_type(default)
    with open(config_location) as f:
        config = json.load(f)
    return expected_type(config.get(key, default))

def get_config_int(key: str, default, doc, obj = None) -> int:
    return get_config(key, default, doc, expected_type=int, obj=obj)

def get_config_str(key: str, default, doc, obj = None) -> str:
    return get_config(key, default, doc, expected_type=str, obj=obj)

def get_config_bool(key: str, default, doc, obj = None) -> bool:
    return get_config(key, default, doc, expected_type=bool, obj=obj)

def get_config_list(key: str, default, doc, obj = None) -> list:
    res =  get_config(key, default, doc, expected_type=list, obj=obj)
    if res is None or res == "" or len(res) == 0 or res == [""]:
        return default
    return res

def get_config_dict(key: str, default, doc, obj=None) -> dict:
    return get_config(key, default, doc, expected_type=dict, obj=obj)

def get_config_float(key: str, default, doc, obj=None) -> float:
    return get_config(key, default, doc, expected_type=float, obj=obj)

def set_config(key: str, value):
    config_location = get_config_location()
    config = {}
    if os.path.exists(config_location):
        with open(config_location) as f:
            config = json.load(f)
    config[key] = value
    with open(config_location, "w") as f:
        json.dump(config, f)

def get_search_path():
    return get_config_list("plugin_search_path", [
        "/usr/local/share/" + PROJECT_NAME + "/plugins",
        "/usr/share/" + PROJECT_NAME + "/plugins",
        os.path.join(os.path.expanduser("~"), "." + PROJECT_NAME, "plugins"),
        os.path.join(os.path.expanduser("~"), ".local", "share", PROJECT_NAME, "plugins"),
        os.getcwd()
    ], "List of plugin search paths")

get_disabled_plugins = lambda: get_config_list("disabled_plugins", [], "List of plugins to disable")

import importlib
import pkgutil

def _discover_submodules(package_name):
    """
    Given a string like 'my_package', discover and import all submodules of 'my_package'
    even if they're not explicitly imported in my_package/__init__.py.
    """
    # First, import the base package
    package = importlib.import_module(package_name)

    # If the package has a __path__ attribute, we can walk its submodules
    # (Packages almost always have __path__, unless it's a namespace pkg).
    if not hasattr(package, '__path__'):
        # It's not a package or does not have submodules in a normal sense
        return []

    submodules = []
    for finder, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        submodules.append(modname)

    return submodules

def _expand_plugin(plugin):
    if plugin.endswith(".*"):
        return [name for name in _discover_submodules(plugin[:-2])]
    return [plugin]

def list_plugins():
    plugins = DEFAULT_PLUGINS
    plugins += get_config_list("plugins", [], "List of plugins to load")
    disabled_plugins = get_disabled_plugins()
    plugins = [plugin for p in plugins for plugin in _expand_plugin(p)]
    plugins = [plugin for plugin in plugins if plugin not in disabled_plugins]
    return list(set(plugins))

def load_plugins():
    oldpath = sys.path
    sys.path = get_search_path() + sys.path
    plugins = list_plugins()
    result = []
    for plugin in plugins:
        try:
            result.append((plugin, importlib.import_module(plugin)))
        except ImportError as e:
            print(f"Error loading plugin {plugin}: {e}")
    sys.path = oldpath
    return result

def notify_plugins(method_name, *args, **kwargs):
    plugins = load_plugins()
    for name, plugin in plugins:
        if hasattr(plugin, method_name):
            getattr(plugin, method_name)(*args, **kwargs)
    

def get_plugin_result(method_name, *args, **kwargs):
    plugins = load_plugins()
    found_results = []
    for name, plugin in plugins:
        if hasattr(plugin, method_name):
            found_results.append((name, getattr(plugin, method_name)(*args, **kwargs)))
    if len(found_results) == 0:
        return None
    if len(found_results) == 1:
        return found_results[0][1]
    raise ConflictError(f"Multiple plugins returned results for {method_name}: {found_results}")

def get_symbols_satisfying(predicate):
    plugins = load_plugins()
    result = []
    visited = set()
    for name, plugin in plugins:
        for symbol in dir(plugin):
            if (name, symbol) in visited:
                continue
            visited.add((name, symbol))
            if predicate(getattr(plugin, symbol)):
                result.append(getattr(plugin, symbol))
    return result


def get_classes_inheriting(base_class):
    return get_symbols_satisfying(lambda x: isinstance(x, type) and issubclass(x, base_class) and x != base_class)



def pluggable(func):
    func_name = func.__name__
    def wrapper(*args, **kwargs):
        result = get_plugin_result(func_name, *args, **kwargs)
        if result is not None:
            return result
        return func(*args, **kwargs)
    return wrapper