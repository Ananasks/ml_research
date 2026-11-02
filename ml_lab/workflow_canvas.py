"""Inline Streamlit v2 component for the interactive experiment canvas."""
from pathlib import Path

import streamlit as st

_ASSET_DIR = Path(__file__).with_name('workflow_canvas_frontend')
_HTML = (_ASSET_DIR / 'canvas.html').read_text(encoding='utf-8')
_CSS = (_ASSET_DIR / 'canvas.css').read_text(encoding='utf-8')
_JS = (_ASSET_DIR / 'canvas.js').read_text(encoding='utf-8')

_render_canvas = st.components.v2.component(
    'ml_research_workflow_canvas_v2', html=_HTML, css=_CSS, js=_JS,
)


def workflow_canvas(nodes, edges, selected=None, height=640, key=None):
    """Render a draggable visual graph and return graph edits from the browser."""
    state = _render_canvas(
        data={'nodes': nodes, 'edges': edges, 'selected': selected, 'height': height},
        key=key,
        on_value_change=lambda: None,
    )
    value = getattr(state, 'value', None)
    if isinstance(value, dict):
        return value
    return None
