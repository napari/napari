from pathlib import Path

from jinja2 import Template
from pydantic import BaseModel
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from napari._qt.dialogs.preferences_dialog import PreferencesDialog
from napari._qt.qt_event_loop import get_app
from napari._qt.qt_resources import get_stylesheet
from napari.settings import NapariSettings

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
GUIDES_PATH = REPO_ROOT_PATH / "docs" / "guides"
IMAGES_PATH = REPO_ROOT_PATH / "docs" / "images" / "_autogenerated"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
PREFERENCES_TEMPLATE = """(preferences)=

# Preferences

Starting with version 0.4.6, napari provides persistent settings.

Settings are managed by the global `SETTINGS` object and can be imported as:

```python
from napari.settings import SETTINGS
```

## Sections

The settings are grouped by sections and napari core provides the following:

{%- for section, section_data in sections.items() %}

### {{ section_data["title"]|upper }}

{{ section_data["description"] }}

{%   for fields in section_data["fields"] %}
#### {{ fields["title"] }}

*{{ fields["description"] }}*

* <small>Access programmatically with `SETTINGS.{{ section }}.{{ fields["field"] }}`.</small>
* <small>Type: `{{ fields["type"] }}`.</small>
* <small>Default: `{{ fields["default"] }}`.</small>
{% if fields["ui"] %}* <small>UI: This setting can be configured via the preferences dialog.</small>{% endif %}
{%-   endfor -%}
{% endfor %}

**Support for plugin specific settings will be provided in an upcoming release.**

## Changing settings programmatically

```python
from napari.settings import SETTINGS

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

![{{ section }}]({{ images_path }}preferences-{{ section }}.png)

{% endfor%}

### Reset to defaults via UI

To reset the preferences click on the `Restore defaults` button and continue
by clicking on `Restore`.

![{{ reset }}]({{ images_path }}preferences-reset.png)

"""


def generate_images():
    """
    Generate images from `CORE_SETTINGS`. and save them in the developer
    section of the docs.
    """

    app = get_app()
    pref = PreferencesDialog()
    pref.setStyleSheet(get_stylesheet("dark"))
    pref.show()
    QTimer.singleShot(1000, pref.close)

    for idx, (name, field) in enumerate(NapariSettings.__fields__.items()):
        pref.set_current_index(idx)
        pixmap = pref.grab()
        title = field.field_info.title or name
        pixmap.save(str(IMAGES_PATH / f"preferences-{title.lower()}.png"))

    box = QMessageBox(
        QMessageBox.Icon.Question,
        "Restore Settings",
        "Are you sure you want to restore default settings?",
        QMessageBox.RestoreDefaults | QMessageBox.Cancel,
        pref,
    )
    box.show()

    def grab():
        pixmap = box.grab()
        pixmap.save(str(IMAGES_PATH / "preferences-reset.png"))
        box.reject()

    QTimer.singleShot(300, grab)
    app.exec_()


def create_preferences_docs():
    """Create preferences docs from SETTINGS using a jinja template."""
    sections = {}

    for name, field in NapariSettings.__fields__.items():

        if not issubclass(field.type_, BaseModel):
            continue

        excluded = getattr(field.type_.NapariConfig, "preferences_exclude", [])
        title = field.field_info.title or name
        sections[title.lower()] = {
            "title": title,
            "description": field.field_info.description or '',
            "fields": [
                {
                    "field": n,
                    "title": f.field_info.title,
                    "description": f.field_info.description,
                    "default": repr(f.get_default()),
                    "ui": n not in excluded,
                    "type": repr(f._type_display()).replace('.typing', ''),
                }
                for n, f in sorted(field.type_.__fields__.items())
                if n not in ('schema_version')
            ],
        }

    text = Template(PREFERENCES_TEMPLATE).render(
        sections=sections, images_path="../images/_autogenerated/"
    )
    (GUIDES_PATH / "preferences.md", "w").write_text(text)


if __name__ == "__main__":
    generate_images()
    create_preferences_docs()
