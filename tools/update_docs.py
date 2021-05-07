from pathlib import Path

from jinja2 import Template
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from napari._qt.dialogs.preferences_dialog import PreferencesDialog
from napari._qt.qt_resources import get_stylesheet
from napari.utils.settings._defaults import CORE_SETTINGS

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
GUIDES_PATH = REPO_ROOT_PATH / "docs" / "guides"
IMAGES_PATH = GUIDES_PATH / "images"
PREFERENCES_TEMPLATE = """(preferences)=

# Preferences

Starting with version 0.4.6, napari provides persistent settings.

Settings are managed by the global `SETTINGS` object and can be imported as:

```python
from napari.utils.settings import SETTINGS
```

## Sections

The settings are grouped by sections and napari core provides the following:

{%- for section, section_data in sections.items() %}

### {{ section_data["title"] }}

{{ section_data["description"] }}

{%   for fields in section_data["fields"] %}
#### {{ fields["title"] }}

{{ fields["description"] }}

* Access programmatically with `SETTINGS.{{ section }}.{{ fields["field"] }}`.
* Type: `{{ fields["type"] }}`.
* Default: `{{ fields["default"] }}`.
{% if fields["ui"] %}* UI: This setting can be configured via the preferences dialog.{% endif %}
{%-   endfor -%}
{% endfor %}

**Support for plugin specific settings will be provided in an upcoming release.**

## Changing settings programmatically

```python
from napari.utils.settings import SETTINGS

SETTINGS.appearance.theme = "light"
```

## Reset to defaults via CLI

To reset all napari settings to the default values:

```bash
napari --reset
```

## The preferences dialog

Starting with version 0.4.6, napari provides a preferences dialog to manage
some of the provided options.

{%- for section, section_data in sections.items() %}

### {{ section_data["title"] }}

![{{ section }}](images/preferences-{{ section }}.png)

{% endfor%}

### Reset to defaults via UI

To reset the preferences click on the `Restore defaults` button and continue
by clicking on `Restore`.

![{{ reset }}](images/preferences-reset.png)

"""


def generate_images():
    """
    Generate images from `CORE_SETTINGS`. and save them in the developer
    section of the docs.
    """
    app = QApplication([])
    pref = PreferencesDialog()
    pref.setStyleSheet(get_stylesheet("dark"))

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(pref.close)
    timer.start(2000)

    pref.show()

    for idx, setting in enumerate(CORE_SETTINGS):
        schema = setting.schema()
        title = schema.get("title").lower()
        pref.set_current_index(idx)
        pixmap = pref.grab()
        pixmap.save(str(IMAGES_PATH / f"preferences-{title}.png"))

    def grab():
        pixmap = pref._reset_dialog.grab()
        pixmap.save(str(IMAGES_PATH / "preferences-reset.png"))
        pref._reset_dialog.close()

    timer2 = QTimer()
    timer2.setSingleShot(True)
    timer2.timeout.connect(grab)
    timer2.start(300)

    pref.restore_defaults()

    app.exec_()


def create_preferences_docs():
    """Create preferences docs from SETTINGS using a jinja template."""
    sections = {}
    for setting in CORE_SETTINGS:
        schema = setting.schema()
        title = schema.get("title", "")
        description = schema.get("description", "")

        section = title.lower()
        sections[section] = {
            "title": title,
            "description": description,
            "fields": [],
        }

        preferences_exclude = getattr(
            setting.NapariConfig, "preferences_exclude", []
        )

        schema = setting.__fields__
        for field in sorted(setting.__fields__):
            if field not in ["schema_version"]:
                data = schema[field].field_info
                default = repr(schema[field].get_default())
                title = data.title
                description = data.description
                try:
                    type_ = schema[field].type_
                except Exception:
                    pass

                sections[section]["fields"].append(
                    {
                        "field": field,
                        "title": title,
                        "description": description,
                        "default": default,
                        "ui": field not in preferences_exclude,
                        "type": type_,
                    }
                )

    template = Template(PREFERENCES_TEMPLATE)
    text = template.render(sections=sections)

    with open(GUIDES_PATH / "preferences.md", "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    generate_images()
    create_preferences_docs()
