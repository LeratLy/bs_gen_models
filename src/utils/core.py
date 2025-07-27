import importlib

def get_class(class_str):
    """
    Get class for full loading string (dynamically import)
    """
    module_name, class_name = class_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def save_call(obj, method, *args, **kwargs):
    """
    Check dynamically if a method exists for the object and call it with corresponding parameters
    """
    if hasattr(obj, method) and callable(func := getattr(obj, method)):
        if not hasattr(func, '__isabstractmethod__'):
            func(*args, **kwargs)

def is_callable(obj, method):
    func = None
    _is_callable = hasattr(obj, method) and callable(func := getattr(obj, method))
    return _is_callable and func is not None and not hasattr(func, '__isabstractmethod__')