from vispy.color import get_color_dict
from napari.utils.colormaps.standardize_color import hex_to_name


def test_hex_to_name_is_updated():
    fail_msg = (
        "If this test fails then vispy have probably updated their color dictionary, located "
        "in vispy.color.get_color_dict. This not necessarily a bad thing, but make sure that "
        "nothing terrible has happened due to this change."
    )
    new_hex_to_name = {
        f"{v.lower()}ff": k for k, v in get_color_dict().items()
    }
    new_hex_to_name["#00000000"] = 'transparent'
    assert new_hex_to_name == hex_to_name, fail_msg
