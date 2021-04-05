(translations)=

# Translations

Starting with version 0.4.7, napari codebase include internationalization
(i18n) and now offers the possibility of installing language packs, which
provide localization (l10n) enabling the user interface to be displayed in
different languages.

To learn more about the current languages that are in the process of
translation, visit the [language packs repository](https://github.com/napari/napari-language-packs)

This guide is limited to providing translations to the napari core codebase.
We will soon provide more information on how to make your napari plugins 
localizable.

## How to make strings translatable?

To make your code translatable (localizable), please use the `trans` helper
provided by the napari utilities.

```python
from napari.utils.translations import trans
```

`trans` is a convenience wrapper on top of the `gettext` module of the
standard library, which provides the l10n and i18n facilities in Python. To
learn more about `gettext` visit the [Python documentation](https://docs.python.org/3/library/gettext.html).

`trans` provides 3 helper methods that can be used to handle localization.

The following examples make use of a widget to illustrate the workflow, but
this also applies to other strings that may be surfaced to the user, e.g.
custom exceptions.

`f-strings` do not work with localizable strings, so any strings that use them
need to be converted to use `str.format` method. The [Plural strings section](#plural-strings)
below explains this in more detail.

### Singular strings

Strings that need to provide a 1:1 translation, can use the `trans._` method:

```python
from qtpy.QtWidgets import QComboBox, QWidget

from napari.utils.translations import trans


class SomeWidget(QWidget):

    def __init__(self):
        self.channel_combo_box = QComboBox(self)        
        self.channel_combo_box.addItem(trans._("red"), "red")
        self.channel_combo_box.addItem(trans._("green"), "green")
        self.channel_combo_box.addItem(trans._("blue"), "blue")
```

On this example, we add a RGB channel combo box selector. The first argument of 
`addItem` is the actual display name of that combobox item, whereas the second
argument is the data associated to that item, in this case the original channel
name.

For the `English (US)` translation the options displayed would be the same,
since `gettext` uses the source language as the key to find translations.
In this case:

  * red
  * green
  * blue

For the `Spanish (Spain)` translation the options displayed would be:

  * rojo
  * verde
  * azul

`trans._` is a wrapper on top of [gettext.gettext](https://docs.python.org/3/library/gettext.html#gettext.gettext).

### Singular strings with context

Strings that need some additional context to disambiguate the source string,
can use the `trans._p` method:

The word **Tab** can mean different things in the english language:
  * A spacer, when the `Tab` key of a keyboard is used inside a text editor.
  * A tablature, a simplified version of sheet music used for stringed
    insruments.
  * A user interface graphical element, like the one provide by `QTabWidget`.

```python
from qtpy.QtWidgets import QComboBox, QWidget

from napari.utils.translations import trans


class SomeWidget(QWidget):

    def __init__(self):
        self.context_combo_box = QComboBox(self)        
        self.context_combo_box.addItem(trans._p("character", "tab", ), "tab")
        self.context_combo_box.addItem(trans._p("music", "tab"), "tab")
        self.context_combo_box.addItem(trans._p("ui-element", "tab"), "tab")
```

On this example, we add the word `tab` three time to a combo box selector.
The first argument of `trans._p` provides the context string that will help
to disambiguate the translation.

For the `English (US)` translation the options displayed would be the same,
since `gettext` uses the source language as the key to find translations. In
this case:

  * tab
  * tab
  * tab

For the `Spanish (Spain)` translation the options displayted would be:

  * tabulación
  * tablatura
  * pestaña

`trans._p` is a wrapper on top of [gettext.pgettext](https://docs.python.org/3/library/gettext.html#gettext.pgettext).

### Plural strings

Some strings or sentences might need to be handled differently when needing
pluralization, that is depending on the amount of items included in the
string.

```python
from qtpy.QtWidgets import QLabel, QWidget

from napari.utils.translations import trans


class SomeWidget(QWidget):

    def __init__(self, amount):
        string = trans._n(
            "{amount} item", "{amount} items", amount
        ).format(amount=amount)
        self.label = QLabel(string)
```

On this example, the label string depends on the `amount` parameter. The
first argument of `trans._n` provides the singular version of the string.
The second argument provides the plural version of the string. The third
argument provides the quantity that will allow to know which string should
be used, in this case `amount`.

For the `English (US)` translation the string displayed for different values
of `amount` would be:
  * For `amount=0`, `"0 items"`
  * For `amount=1`, `"1 item"`
  * For `amount=2`, `"2 items"`

For the `Spanish (Spain)` translation the options displayted would be:
  * For `amount=0`, `"0 ítems"`
  * For `amount=1`, `"1 ítem"`
  * For `amount=2`, `"2 ítems"`

Take into account that different languages will handle pluralization
differently. Having clear variable names within strings (e.g. `{amount}`) of
what the variable represents makes the internationalization process much
easier and pleasant for translators.

`trans._n` is a wrapper on top of [gettext.ngettext](https://docs.python.org/3/library/gettext.html#gettext.ngettext).

## Contributing translations

To be able to provide translations for different languages, the napari team
chose to use the `Crowdin` which provides a simple web interface where the
localizable strings can be translated one by one with the help of volunteers
and the napari user community.

To start translating, please visit the [Crowdin project page](https://crowdin.com/project/napari).
