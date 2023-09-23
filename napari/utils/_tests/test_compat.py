from napari.utils.compat import StrEnum


def test_str_enum():
    class Cake(StrEnum):
        CHOC = "chocolate"
        VANILLA = "vanilla"
        STRAWBERRY = "strawberry"

    assert Cake.CHOC == "chocolate"
    assert Cake.CHOC == Cake.CHOC
    assert f'{Cake.CHOC}' == "chocolate"
    assert Cake.CHOC != "vanilla"
    assert Cake.CHOC != Cake.VANILLA
