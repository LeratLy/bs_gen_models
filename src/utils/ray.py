from ray import train as ray_train

def is_ray_running():
    return ray_train._internal.session.get_session() is not None
