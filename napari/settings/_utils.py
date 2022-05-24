def _coerce_extensions_to_globs(reader_settings):
    """Coerce existing reader settings for file extensions to glob patterns"""
    new_settings = {}
    for pattern, reader in reader_settings.items():
        if pattern.startswith('.') and '*' not in pattern:
            pattern = f"*{pattern}"
        new_settings[pattern] = reader
    return new_settings
