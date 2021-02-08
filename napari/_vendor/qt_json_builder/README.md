# qt-jsonschema-form
A tool to generate Qt forms from JSON Schemas. 

## Features
* Error messages from JSONSchema validation ([see jsonschema](https://github.com/Julian/jsonschema)).
* Widgets for file selection, colour picking, date-time selection (and more).
* Per-field widget customisation is provided by an additional ui-schema (inspired by https://github.com/mozilla-services/react-jsonschema-form).

## Unsupported validators
Currently this tool does not support `anyOf` or `oneOf` directives. The reason for this is simply that these validators have different semantics depending upon the context in which they are found. Primitive support could be added with meta-widgets for type schemas.

Additionally, the `$ref` keyword is not supported. This will be fixed, but is waiting on some proposed upstream changes in `jsonschema`

## Example
```python3
import sys
from json import dumps

from PyQt5 import QtWidgets

from qt_jsonschema_form import WidgetBuilder

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    builder = WidgetBuilder()

    schema = {
        "type": "object",
        "title": "Number fields and widgets",
        "properties": {
            "schema_path": {
                "title": "Schema path",
                "type": "string"
            },
            "integerRangeSteps": {
                "title": "Integer range (by 10)",
                "type": "integer",
                "minimum": 55,
                "maximum": 100,
                "multipleOf": 10
            },
            "event": {
                "type": "string",
                "format": "date"
            },
            "sky_colour": {
                "type": "string"
            },
            "names": {
                "type": "array",
                "items": [
                    {
                        "type": "string",
                        "pattern": "[a-zA-Z\-'\s]+",
                        "enum": [
                            "Jack", "Jill"
                        ]
                    },
                    {
                        "type": "string",
                        "pattern": "[a-zA-Z\-'\s]+",
                        "enum": [
                            "Alice", "Bob"
                        ]
                    },
                ],
                "additionalItems": {
                    "type": "number"
                },
            }
        }
    }

    ui_schema = {
        "schema_path": {
            "ui:widget": "filepath"
        },
        "sky_colour": {
            "ui:widget": "colour"
        }

    }
    form = builder.create_form(schema, ui_schema)
    form.widget.state = {
        "schema_path": "some_file.py",
        "integerRangeSteps": 60,
        "sky_colour": "#8f5902",
        "names": [
            "Jack",
            "Bob"
        ]
    }
    form.show()
    form.widget.on_changed.connect(lambda d: print(dumps(d, indent=4)))

    app.exec_()


```
