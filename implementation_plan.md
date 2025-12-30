# Napari Pydantic V2 Migration - Implementation Plan

## Overview

**Goal**: Migrate napari from Pydantic V1 (via `pydantic.v1` compatibility layer) to native Pydantic V2 to enable Python 3.14 support.

**Issue Reference**: https://github.com/napari/napari/issues/8493

**Root Cause**: Napari uses `pydantic.v1` compatibility layer which has compatibility issues with Python 3.14. The specific problems include:
- `__slots__` conflicts with Pydantic model creation
- `ModelMetaclass` implementation conflicts with Python 3.14's class creation changes
- `__get_validators__` protocol issues with Python 3.14's descriptor protocol

---

## Scope Summary

| Metric | Count |
|--------|-------|
| Files importing from `napari._pydantic_compat` | 51 |
| Files using `@validator` decorator | 13 |
| Files using `@root_validator` decorator | 2 |
| Files with `class Config:` patterns | 10 |
| Files accessing `__fields__` directly | 8 |
| Files with custom `__get_validators__` | 8 |
| Files with `__slots__` (conflict risk) | 2 |
| Files using `BaseSettings` | 3 |

---

## Migration Phases

### Phase 1: Core Infrastructure
**Status**: [ ] Not Started

#### 1.1 Update `_pydantic_compat.py`
- [ ] Create new compatibility shim that imports from native Pydantic V2
- [ ] Map V1 names to V2 equivalents where possible
- [ ] Add deprecation warnings for removed APIs

**V1 to V2 Import Mapping**:
```python
# V1 (current)                    # V2 (target)
from pydantic.v1 import BaseModel  -> from pydantic import BaseModel
from pydantic.v1 import Field      -> from pydantic import Field
from pydantic.v1 import validator  -> from pydantic import field_validator
from pydantic.v1 import root_validator -> from pydantic import model_validator
from pydantic.v1 import PrivateAttr -> from pydantic import PrivateAttr
from pydantic.v1.main import ModelMetaclass -> REMOVED (use __pydantic_complete__)
from pydantic.v1.fields import ModelField -> from pydantic.fields import FieldInfo
```

#### 1.2 Migrate `evented_model.py` (CRITICAL)
**File**: `src/napari/utils/events/evented_model.py`

- [ ] Replace `ModelMetaclass` with V2 model hooks
- [ ] Update `__fields__` access to `model_fields`
- [ ] Fix `__slots__` conflict (line 184)
- [ ] Update `field.field_info.allow_mutation` checks
- [ ] Convert `class Config:` to `model_config = ConfigDict(...)`
- [ ] Update `json_encoders` to `model_serializer`

**Key Changes**:
```python
# Before (V1)
class EventedMetaclass(ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        for n, f in cls.__fields__.items():
            cls.__eq_operators__[n] = pick_equality_operator(f.type_)

# After (V2)
# Use __pydantic_complete__ hook or model_post_init
def model_post_init(self, __context):
    # Initialize eq_operators after model creation
    pass
```

#### 1.3 Migrate `_base.py` (EventedSettings)
**File**: `src/napari/settings/_base.py`

- [ ] Update `BaseSettings` to V2 version
- [ ] Migrate `customise_sources()` to V2 settings customizer
- [ ] Update `__fields__` introspection to `model_fields`
- [ ] Update `field.field_info.extra` access patterns

---

### Phase 2: Validators Migration
**Status**: [ ] Not Started

#### 2.1 @validator to @field_validator

**Files requiring migration** (13 files):

| File | Validators | Status |
|------|-----------|--------|
| `src/napari/components/dims.py` | 6 | [ ] |
| `src/napari/layers/utils/color_manager.py` | 3 | [ ] |
| `src/napari/layers/utils/text_manager.py` | 1 | [ ] |
| `src/napari/layers/utils/plane.py` | 1 | [ ] |
| `src/napari/layers/utils/color_encoding.py` | 1 | [ ] |
| `src/napari/settings/_shortcuts.py` | 1 | [ ] |
| `src/napari/components/camera.py` | 1 | [ ] |
| `src/napari/components/viewer_model.py` | 1 | [ ] |
| `src/napari/components/overlays/base.py` | 1 | [ ] |
| `src/napari/utils/colormaps/colormap.py` | 1 | [ ] |
| `src/napari/utils/colormaps/standardize_color.py` | 1 | [ ] |
| `src/napari/layers/_layer_actions.py` | 1 | [ ] |
| `src/napari/components/layerlist.py` | 1 | [ ] |

**Migration Pattern**:
```python
# Before (V1)
@validator('field_name', pre=True, always=True)
def validate_field(cls, v):
    return v

# After (V2)
@field_validator('field_name', mode='before')
@classmethod
def validate_field(cls, v):
    return v
```

#### 2.2 @root_validator to @model_validator

**Files requiring migration** (2 files):

| File | Root Validators | Status |
|------|----------------|--------|
| `src/napari/components/dims.py` | 1 | [ ] |
| `src/napari/layers/utils/color_manager.py` | 1 | [ ] |

**Migration Pattern**:
```python
# Before (V1)
@root_validator(skip_on_failure=True, allow_reuse=True)
def validate_all(cls, values):
    return values

# After (V2)
@model_validator(mode='after')
def validate_all(self) -> Self:
    return self
```

---

### Phase 3: Config Classes Migration
**Status**: [ ] Not Started

**Files requiring migration** (10 files):

| File | Config Options | Status |
|------|---------------|--------|
| `evented_model.py` | arbitrary_types, validate_assignment, json_encoders | [ ] |
| `_base.py` | env_prefix, customise_sources | [ ] |
| `_napari_settings.py` | Multiple settings | [ ] |
| `_application.py` | Settings config | [ ] |
| `_shortcuts.py` | Settings config | [ ] |
| `_plugins.py` | Settings config | [ ] |
| `debugging.py` | BaseSettings config | [ ] |
| `style_encoding.py` | Model config | [ ] |
| `_source.py` | Model config | [ ] |
| `_appearance.py` | Settings config | [ ] |

**Migration Pattern**:
```python
# Before (V1)
class MyModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        json_encoders = {np.ndarray: lambda arr: arr.tolist()}

# After (V2)
from pydantic import ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    # json_encoders -> use @field_serializer
    @field_serializer('array_field')
    def serialize_array(self, v: np.ndarray) -> list:
        return v.tolist()
```

---

### Phase 4: Custom Types Migration
**Status**: [ ] Not Started

#### 4.1 __get_validators__ to Annotated types

**Files requiring migration** (8 files):

| File | Custom Types | Status |
|------|-------------|--------|
| `src/napari/settings/_fields.py` | Theme, Language | [ ] |
| `src/napari/utils/color.py` | ColorValue, ColorArray | [ ] |
| `src/napari/layers/utils/color_manager.py` | ColorProperties | [ ] |
| `src/napari/utils/colormaps/categorical_colormap.py` | CategoricalColormap | [ ] |
| `src/napari/utils/events/custom_types.py` | Array, ConstrainedInt, ConstrainedFloat | [ ] |

**Migration Pattern**:
```python
# Before (V1)
class CustomType:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v

# After (V2)
from typing import Annotated
from pydantic import BeforeValidator

def validate_custom(v):
    return v

CustomType = Annotated[BaseType, BeforeValidator(validate_custom)]
```

---

### Phase 5: Method Calls Update
**Status**: [ ] Not Started

**Files to update** (~30 files):

| Old Method | New Method | Count |
|-----------|-----------|-------|
| `.dict()` | `.model_dump()` | ~50 |
| `.json()` | `.model_dump_json()` | ~10 |
| `.construct()` | `.model_construct()` | ~5 |
| `.copy()` | `.model_copy()` | ~15 |
| `.parse_obj()` | `.model_validate()` | ~5 |
| `parse_obj_as()` | Use TypeAdapter | ~5 |
| `schema_json()` | `.model_json_schema()` | ~3 |

**Note**: Many of these can be handled with search-and-replace, but require testing.

---

### Phase 6: Settings System Migration
**Status**: [ ] Not Started

#### 6.1 BaseSettings Migration

**Files**: `_base.py`, `debugging.py`

```python
# Before (V1)
from pydantic.v1 import BaseSettings
from pydantic.v1.env_settings import EnvSettingsSource

class MySettings(BaseSettings):
    class Config:
        env_prefix = 'NAPARI_'

    @classmethod
    def customise_sources(cls, init_settings, env_settings, file_secret_settings):
        return (init_settings, env_settings)

# After (V2)
from pydantic_settings import BaseSettings, SettingsConfigDict

class MySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='NAPARI_')

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, ...):
        return (init_settings, env_settings)
```

**Note**: Pydantic V2 splits `BaseSettings` into separate `pydantic-settings` package.

---

## Testing Strategy

### Unit Tests to Update

| Test File | Status |
|-----------|--------|
| `napari/_tests/test_evented_model.py` | [ ] |
| `napari/settings/_tests/test_settings.py` | [ ] |
| `napari/components/_tests/test_dims.py` | [ ] |
| `napari/components/_tests/test_camera.py` | [ ] |
| `napari/layers/_tests/test_color_manager.py` | [ ] |

### Integration Tests

- [ ] Test on Python 3.10 (ensure backward compatibility)
- [ ] Test on Python 3.11
- [ ] Test on Python 3.12
- [ ] Test on Python 3.13
- [ ] Test on Python 3.14 (primary goal)
- [ ] Test on Python 3.14 free-threaded build

---

## Dependencies

### Required Package Updates

```toml
# pyproject.toml changes
dependencies = [
    "pydantic>=2.0",           # Was: pydantic>=2.2.0 (using v1 compat)
    "pydantic-settings>=2.0",  # NEW: Required for BaseSettings
]
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking API changes for plugins | HIGH | Provide migration guide, deprecation warnings |
| EventedModel metaclass complexity | HIGH | Extensive testing, consider alternative approaches |
| Settings serialization changes | MEDIUM | Test config file round-trip |
| Performance regression | LOW | Benchmark critical paths |
| __slots__ conflicts | HIGH | May need to remove __slots__ or use workaround |

---

## Progress Tracking

### Overall Progress: 0%

| Phase | Progress | Notes |
|-------|----------|-------|
| Phase 1: Core Infrastructure | 0% | |
| Phase 2: Validators | 0% | |
| Phase 3: Config Classes | 0% | |
| Phase 4: Custom Types | 0% | |
| Phase 5: Method Calls | 0% | |
| Phase 6: Settings System | 0% | |
| Testing | 0% | |

---

## Implementation Order

1. **Start with `_pydantic_compat.py`** - Create V2-native imports
2. **Migrate `evented_model.py`** - Core infrastructure
3. **Migrate `_base.py`** - Settings infrastructure
4. **Update validators** in order of dependency
5. **Update custom types**
6. **Update method calls** (can be parallelized)
7. **Full test suite** on all Python versions

---

## Notes

- Keep V1 compat shim temporarily for transition period
- Consider feature flag for gradual rollout
- Document all breaking changes for plugin authors
- Coordinate with napari community on migration timeline

---

*Last Updated: 2025-12-30*
*Author: Claude Code (assisting derekthirstrup)*
