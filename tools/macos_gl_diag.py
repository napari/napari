"""Temporary diagnostic for black label screenshots on macOS 26 runners.

Mimics test_label_colors_matching_widget_auto and records, for every frame
where the middle pixel is plain background, the GL state and what repeated
grabs return. Failing frames are saved as PNG to the output directory.

Usage: python tools/macos_gl_diag.py OUT_DIR [--rounds N] [--dtype int16]
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from qtpy import API_NAME, QT_VERSION
from qtpy.QtCore import QCoreApplication, Qt
from qtpy.QtWidgets import QApplication

parser = argparse.ArgumentParser()
parser.add_argument('out', type=Path)
parser.add_argument('--rounds', type=int, default=3)
parser.add_argument('--dtype', default='int16')
parser.add_argument('--no-share', action='store_true')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

if args.no_share:
    QCoreApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts, False
    )

from vispy.gloo import gl  # noqa: E402

from napari._qt.qt_event_loop import get_qapp  # noqa: E402
from napari._qt.qt_viewer import QtViewer  # noqa: E402
from napari.components import ViewerModel  # noqa: E402

app = get_qapp()
if args.no_share:
    QCoreApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts, False
    )

gc_events = []
gc.callbacks.append(
    lambda phase, info: (
        phase == 'stop' and gc_events.append(info['generation'])
    )
)

model = ViewerModel()
qv = QtViewer(model)
qv.show()
qv.resize(460, 460)
QApplication.processEvents()
qv.controls.show()
QApplication.processEvents()
native = qv.canvas.native

native.makeCurrent()
info = {
    'qt_api': API_NAME,
    'qt': QT_VERSION,
    'share_attr': QCoreApplication.testAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    ),
    'vendor': gl.glGetParameter(gl.GL_VENDOR),
    'renderer': gl.glGetParameter(gl.GL_RENDERER),
    'version': gl.glGetParameter(gl.GL_VERSION),
    'format': str(native.format().profile()),
    'dpr': native.devicePixelRatio(),
    'max_tex': int(gl.glGetParameter(gl.GL_MAX_TEXTURE_SIZE)),
}
print(json.dumps(info, default=str), flush=True)

dtype = np.dtype(args.dtype)
layer = model.add_labels(np.ones((2, 2), dtype=dtype))
layer.opacity = 1.0
QApplication.processEvents()


def mid(img):
    return img[tuple(np.array(img.shape[:2]) // 2)]


def gl_state():
    native.makeCurrent()
    return {
        'viewport': [int(v) for v in gl.glGetParameter(gl.GL_VIEWPORT)],
        'fbo': int(gl.glGetParameter(gl.GL_FRAMEBUFFER_BINDING)),
        'default_fbo': int(native.defaultFramebufferObject()),
        'err': int(gl.glGetError()),
    }


def describe(img):
    nonbg = np.any(img[..., :3] != 0, axis=-1)
    ys, xs = np.nonzero(nonbg)
    bbox = (
        [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
        if len(ys)
        else None
    )
    return {
        'shape': list(img.shape),
        'nonbg_frac': round(float(nonbg.mean()), 4),
        'bbox': bbox,
        'n_colors': len(np.unique(img.reshape(-1, 4), axis=0)),
    }


rng = np.random.default_rng(0)
n_c = len(layer.colormap)
iinfo = np.iinfo(dtype)
labels = np.concatenate(
    (
        np.arange(1, 10, dtype=dtype),
        [n_c - 1, n_c, n_c + 1],
        rng.integers(1, min(2**20, iinfo.max), size=20, dtype=dtype),
        [-1, -2, -10] if iinfo.min < 0 else [],
    )
).astype(dtype)

total = 0
failures = 0
from napari._qt.utils import QImg2array  # noqa: E402

for rnd in range(args.rounds):
    for selected in (True, False):
        layer.show_selected_label = selected
        for label in labels:
            total += 1
            gc_events.clear()
            layer.data = np.full((2, 2), label, dtype=dtype)
            layer.selected_label = label
            QApplication.processEvents()
            tex = qv.canvas.layer_to_visual[layer].node.shared_program[
                'texture2D_values'
            ]
            img = qv.screenshot(flash=False, size=[400, 400])
            px = mid(img)
            if px[:3].any():
                continue
            failures += 1
            name = f'r{rnd}_{"sel" if selected else "all"}_{label}'
            from PIL import Image

            Image.fromarray(img).save(args.out / f'{name}_first.png')
            rec = {
                'label': int(label),
                'selected': selected,
                'gc_gens': list(gc_events),
                'tex_id': id(tex),
                'first': describe(img),
                'state_after_first': gl_state(),
            }
            # Same size, no changes: does a plain re-grab render correctly?
            regrab = QImg2array(native.grabFramebuffer())
            rec['regrab_mid'] = mid(regrab).tolist()
            rec['regrab'] = describe(regrab)
            again = qv.screenshot(flash=False, size=[400, 400])
            rec['again_mid'] = mid(again).tolist()
            QApplication.processEvents()
            later = qv.screenshot(flash=False, size=[400, 400])
            rec['after_events_mid'] = mid(later).tolist()
            Image.fromarray(again).save(args.out / f'{name}_again.png')
            print('FAIL', json.dumps(rec), flush=True)

print(f'SUMMARY {failures}/{total} black frames', flush=True)
qv.close()
sys.exit(0)
