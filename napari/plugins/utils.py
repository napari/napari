from . import _npe2, plugin_manager

def get_potential_readers(filename):
    readers = _npe2.get_readers(filename)

    npe1_readers = []
    for spec, hook_caller in plugin_manager.hooks.items():
        if spec == 'napari_get_reader':
            potential_readers = hook_caller.get_hookimpls()
            for get_reader in potential_readers:
                reader = hook_caller._call_plugin(get_reader.plugin_name, path=filename)
                if callable(reader):
                    npe1_readers.append(get_reader.plugin_name)

    return readers, npe1_readers
