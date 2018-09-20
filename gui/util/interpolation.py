from vispy import scene as _scene


# get available interpolation methods
interpolation_names = _scene.visuals.Image(None).interpolation_functions
interpolation_names = list(interpolation_names)
interpolation_names.sort()
# interpolation_names.remove('sinc')  # does not work well on my machine

interpolation_index_to_name = interpolation_names.__getitem__
interpolation_name_to_index = interpolation_names.index
