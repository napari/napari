import csv
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from napari.utils.io import imsave
from napari.utils.misc import abspath_or_url

if TYPE_CHECKING:
    from napari.types import FullLayerData

# 常量定义
CSV_EXTENSION = '.csv'
IMAGE_EXTENSIONS = (
    '.bmp',
    '.bsdf',
    '.bw',
    '.eps',
    '.gif',
    '.icns',
    '.ico',
    '.im',
    '.j2c',
    '.j2k',
    '.jfif',
    '.jp2',
    '.jpc',
    '.jpe',
    '.jpeg',
    '.jpf',
    '.jpg',
    '.jpx',
    '.lsm',
    '.mpo',
    '.npz',
    '.pbm',
    '.pcx',
    '.pgm',
    '.png',
    '.ppm',
    '.ps',
    '.rgb',
    '.rgba',
    '.sgi',
    '.stk',
    '.tga',
    '.tif',
    '.tiff',
)
DEFAULT_IMAGE_EXT = '.tif'
POINTS_DTYPE = np.uint32
SHAPE_DIMENSION_NAMES = ('index', 'shape-type', 'vertex-index')


def write_csv(
    filename: str | Path,
    data: np.ndarray | list,
    column_names: Optional[list[str]] = None,
) -> None:
    """优化后的 CSV 写入函数，支持路径对象和更健壮的错误处理"""
    filename = Path(filename)
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            if column_names:
                writer.writerow(column_names)
            if isinstance(data, np.ndarray):
                writer.writerows(data.tolist())
            else:
                writer.writerows(data)
    except (OSError, PermissionError) as e:
        raise OSError(f'Error writing to CSV file: {filename}') from e


def imsave_extensions() -> tuple[str, ...]:
    """返回 imsave 支持的图像格式扩展名"""
    return IMAGE_EXTENSIONS


def _ensure_extension(path: str | Path, ext: str) -> Path:
    """确保路径具有指定的扩展名"""
    path = Path(path)
    if not path.suffix:
        return path.with_suffix(ext)
    return path


def _write_image_data(
    path: str | Path, data: Any, meta: dict
) -> Optional[Path]:
    """内部图像写入函数，处理多尺度数据和错误"""
    path = Path(path)

    # 处理多尺度图像
    if meta.get('multiscale', False):
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        written = []
        for i, level in enumerate(data):
            level_path = path / f'level_{i}{DEFAULT_IMAGE_EXT}'
            try:
                imsave(str(level_path), level)
                written.append(level_path)
            except Exception as e:
                warnings.warn(f'Failed to write multiscale level {i}: {e!s}')
        return written[0] if written else None

    # 单图像处理
    path = _ensure_extension(path, DEFAULT_IMAGE_EXT)
    try:
        imsave(str(path), data)
        return path
    except Exception as e:
        warnings.warn(f'Failed to write image: {e!s}')
        return None


def napari_write_image(path: str, data: Any, meta: dict) -> Optional[str]:
    """优化后的图像写入函数，支持多尺度图像和错误处理"""
    result = _write_image_data(path, data, meta)
    return str(result) if result else None


def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    """优化后的标签写入函数，自动处理数据类型"""
    # 确保标签数据使用足够大的整数类型
    if data.dtype.itemsize < 4:
        data = data.astype(POINTS_DTYPE)
    return napari_write_image(path, data, meta)


def _create_points_table(
    data: np.ndarray, properties: dict
) -> tuple[np.ndarray, list[str]]:
    """为点数据创建表格结构和列名"""
    n_points, n_dims = data.shape
    column_names = [f'axis-{d}' for d in range(n_dims)]
    table = data.copy()

    # 添加属性列
    if properties:
        prop_arrays = []
        for name, values in properties.items():
            if len(values) != n_points:
                warnings.warn(
                    f"Property '{name}' has incorrect length, skipping"
                )
                continue
            prop_arrays.append(values)
            column_names.append(name)

        if prop_arrays:
            table = np.column_stack([table, *prop_arrays])

    # 添加索引列
    indices = np.arange(n_points).reshape(-1, 1)
    table = np.column_stack([indices, table])
    column_names = ['index'] + column_names

    return table, column_names


def napari_write_points(path: str, data: Any, meta: dict) -> Optional[str]:
    """优化后的点数据写入函数"""
    path = Path(path)
    path = _ensure_extension(path, CSV_EXTENSION)

    if path.suffix != CSV_EXTENSION:
        return None

    properties = meta.get('properties', {})
    table, column_names = _create_points_table(data, properties)

    try:
        write_csv(path, table, column_names)
        return str(path)
    except Exception as e:
        warnings.warn(f'Failed to write points: {e!s}')
        return None


def _create_shapes_table(
    data: list, shape_types: list
) -> tuple[np.ndarray, list[str]]:
    """为形状数据创建表格结构和列名"""
    # 确定最大维度
    max_dims = max(shape.shape[1] for shape in data)
    column_names = [f'axis-{d}' for d in range(max_dims)]
    column_names = SHAPE_DIMENSION_NAMES + tuple(column_names)

    # 构建表格行
    rows = []
    for shape_idx, (vertices, shape_type) in enumerate(
        zip(data, shape_types, strict=False)
    ):
        n_vertices = vertices.shape[0]
        n_dims = vertices.shape[1]

        # 填充不足的维度
        if n_dims < max_dims:
            padded_vertices = np.pad(
                vertices,
                ((0, 0), (0, max_dims - n_dims)),
                mode='constant',
                constant_values=np.nan,
            )
        else:
            padded_vertices = vertices

        for vertex_idx, vertex in enumerate(padded_vertices):
            rows.append([shape_idx, shape_type, vertex_idx, *vertex.tolist()])

    return np.array(rows), column_names


def napari_write_shapes(path: str, data: Any, meta: dict) -> Optional[str]:
    """优化后的形状数据写入函数"""
    path = Path(path)
    path = _ensure_extension(path, CSV_EXTENSION)

    if path.suffix != CSV_EXTENSION:
        return None

    if not data:
        return None

    shape_types = meta.get('shape_type', ['rectangle'] * len(data))
    if len(shape_types) != len(data):
        shape_types = ['rectangle'] * len(data)
        warnings.warn('Mismatch between shape data and types, using defaults')

    try:
        table, column_names = _create_shapes_table(data, shape_types)
        write_csv(path, table, column_names)
        return str(path)
    except Exception as e:
        warnings.warn(f'Failed to write shapes: {e!s}')
        return None


def write_layer_data_with_plugins(
    path: str, layer_data: list['FullLayerData']
) -> list[str]:
    """优化后的分层数据写入函数，增强错误处理和原子性"""
    path = Path(path)
    already_existed = path.exists()

    # 创建输出目录
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f'Could not create directory {path}: {e!s}') from e

    written_files = []
    try:
        with TemporaryDirectory(dir=path) as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 使用npe2写入每层数据
            import npe2

            for layer in layer_data:
                _, meta, layer_type = layer
                layer_name = meta.get('name', f'layer_{len(written_files)}')
                layer_path = tmp_path / layer_name

                try:
                    npe2.write(
                        path=abspath_or_url(layer_path),
                        layer_data=[layer],
                        plugin_name='napari',
                    )
                    # 收集写入的文件
                    if layer_path.is_file():
                        written_files.append(layer_path)
                    elif layer_path.is_dir():
                        written_files.extend(layer_path.rglob('*'))
                except Exception as e:
                    warnings.warn(
                        f"Failed to write layer '{layer_name}': {e!s}"
                    )

            # 将临时文件移动到最终位置
            for src in written_files:
                dest = path / src.relative_to(tmp_path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dest))
                written_files[written_files.index(src)] = str(dest)

    except Exception:
        # 清理：如果目录原本不存在，则删除整个目录
        if not already_existed:
            shutil.rmtree(path, ignore_errors=True)
        raise

    return [str(p) for p in written_files]
