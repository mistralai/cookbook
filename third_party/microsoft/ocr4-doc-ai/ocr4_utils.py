"""Reusable helpers for the Azure Mistral OCR-4 notebooks."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv
from PIL import Image


BLOCK_COLORS = {
    'title': '#E74C3C',
    'text': '#3498DB',
    'aside_text': '#2ECC71',
    'table': '#E67E22',
    'image': '#9B59B6',
    'list': '#1ABC9C',
    'equation': '#F1C40F',
    'caption': '#E91E63',
    'code': '#795548',
    'references': '#0D47A1',
    'header': '#FF6F00',
    'footer': '#004D40',
    'signature': '#607D8B',
}

PARAGRAPH_TYPES = {'text', 'title', 'list', 'aside_text', 'caption', 'references'}


@dataclass(frozen=True)
class OCRConfig:
    """Connection settings for an Azure-hosted Mistral OCR deployment."""

    endpoint: str
    api_key: str
    model_name: str

    @classmethod
    def from_env(cls) -> 'OCRConfig':
        load_dotenv()
        config = cls(
            endpoint=os.getenv('AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT', ''),
            api_key=os.getenv('AZURE_MISTRAL_DOCUMENT_AI_KEY', ''),
            model_name=os.getenv('AZURE_AI_DEPLOYMENT_NAME', ''),
        )
        config.validate()
        return config

    def validate(self) -> None:
        missing = [
            name for name, value in {
                'AZURE_MISTRAL_DOCUMENT_AI_ENDPOINT': self.endpoint,
                'AZURE_MISTRAL_DOCUMENT_AI_KEY': self.api_key,
                'AZURE_AI_DEPLOYMENT_NAME': self.model_name,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f'Missing required environment variables: {", ".join(missing)}')

    @property
    def headers(self) -> dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }



def sample_path(name: str, samples_dir: str | Path = 'samples') -> Path:
    """Return a path below the repository's samples directory."""
    return Path(samples_dir) / name



def encode_file(file_path: str | Path) -> str:
    """Read a local file and return its base64 representation."""
    return base64.b64encode(Path(file_path).read_bytes()).decode('ascii')



def data_url(file_path: str | Path) -> str:
    """Build a data URL using the file's detected MIME type."""
    path = Path(file_path)
    mime_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return f'data:{mime_type};base64,{encode_file(path)}'



def resolve_config(config: OCRConfig | None = None) -> OCRConfig:
    """Use an explicit connection config or load one from the environment."""
    return config or OCRConfig.from_env()


def ocr_request(
    payload: Mapping[str, Any],
    config: OCRConfig | None = None,
) -> dict[str, Any]:
    """Submit one OCR-4 request and return its JSON response."""
    config = resolve_config(config)
    response = requests.post(
        config.endpoint,
        json=dict(payload),
        headers=config.headers,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()



def build_ocr_payload(
    file_path: str | Path,
    config: OCRConfig | None = None,
    *,
    include_blocks: bool = True,
    confidence_granularity: str | None = 'page',
    **options: Any,
) -> dict[str, Any]:
    """Build a consistent OCR-4 payload for a local document."""
    config = resolve_config(config)
    payload = {
        'model': config.model_name,
        'document': {'type': 'document_url', 'document_url': data_url(file_path)},
        'include_blocks': include_blocks,
    }
    if confidence_granularity:
        payload['confidence_scores_granularity'] = confidence_granularity
    payload.update(options)
    return payload



def call_ocr4(
    file_path: str | Path,
    config: OCRConfig | None = None,
) -> dict[str, Any]:
    """Run OCR-4 with blocks and page-level confidence enabled."""
    config = resolve_config(config)
    payload = build_ocr_payload(file_path, config)
    response = requests.post(
        config.endpoint,
        json=payload,
        headers=config.headers,
        timeout=120,
    )
    if response.status_code == 422:
        response = requests.post(
            config.endpoint,
            json={key: payload[key] for key in ('model', 'document')},
            headers=config.headers,
            timeout=120,
        )
    response.raise_for_status()
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise RuntimeError(f'OCR endpoint returned non-JSON content: {response.text[:500]}') from error



def extract_text(ocr_response: Mapping[str, Any]) -> str:
    """Extract normalized text from OCR markdown or block content."""
    parts: list[str] = []
    for page in ocr_response.get('pages', []):
        if page.get('markdown'):
            parts.append(page['markdown'])
        else:
            parts.extend(
                block.get('content', '')
                for block in page.get('blocks', [])
                if block.get('content')
            )
    text = re.sub(r'\s+', ' ', '\n'.join(parts)).strip()
    if not text:
        raise ValueError('OCR-4 returned no recognizable text.')
    return text



def parse_annotation(value: Any) -> dict[str, Any]:
    """Normalize an annotation returned as a dictionary or JSON string."""
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def build_annotation_payload(
    file_path: str | Path,
    annotation_format: str,
    config: OCRConfig | None = None,
) -> dict[str, Any]:
    """Build the standard image/document annotation request used in examples."""
    config = resolve_config(config)
    return {
        **build_ocr_payload(
            file_path,
            config,
            include_blocks=True,
            confidence_granularity='page',
            include_image_base64=True,
            extract_header=True,
            extract_footer=True,
        ),
        annotation_format: {
            'type': 'json_schema',
            'json_schema': {
                'name': 'image_annotation',
                'description': 'Structured information extracted from an image.',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'image_type': {
                            'type': 'string',
                            'description': 'The type of the image.',
                        },
                        'short_description': {
                            'type': 'string',
                            'description': 'A concise English description of the image.',
                        },
                    },
                },
            },
        },
    }


def bboxannotation(
    file_path_or_encoding: str | Path,
    annotation_format: str,
    config: OCRConfig | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the original post-release-check notebook."""
    if isinstance(file_path_or_encoding, Path):
        return build_annotation_payload(file_path_or_encoding, annotation_format, config)
    if isinstance(file_path_or_encoding, str) and len(file_path_or_encoding) < 256:
        candidate = Path(file_path_or_encoding)
        if candidate.exists():
            return build_annotation_payload(candidate, annotation_format, config)
    config = resolve_config(config)
    return {
        'model': config.model_name,
        'include_blocks': True,
        'confidence_scores_granularity': 'page',
        'include_image_base64': True,
        'extract_header': True,
        'extract_footer': True,
        'document': {
            'type': 'document_url',
            'document_url': f'data:application/pdf;base64,{file_path_or_encoding}',
        },
        annotation_format: {
            'type': 'json_schema',
            'json_schema': {
                'name': 'image_annotation',
                'description': 'Structured information extracted from an image.',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'image_type': {'type': 'string'},
                        'short_description': {'type': 'string'},
                    },
                },
            },
        },
    }



def page_confidence(page: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Return average and minimum confidence for one OCR page."""
    scores = page.get('confidence_scores') or {}
    return (
        scores.get('average_page_confidence_score'),
        scores.get('minimum_page_confidence_score'),
    )



def blocks_to_dataframe(blocks: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert OCR blocks into a stable geometry/classification table."""
    rows = []
    for block in blocks:
        x1, y1 = block['top_left_x'], block['top_left_y']
        x2, y2 = block['bottom_right_x'], block['bottom_right_y']
        rows.append({
            'type': block.get('type', 'unknown'),
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'width': x2 - x1,
            'height': y2 - y1,
            'area': (x2 - x1) * (y2 - y1),
            'content': (block.get('content') or '').replace('\n', ' '),
        })
    return pd.DataFrame(rows)



def plot_bounding_boxes(
    image_path: str | Path,
    blocks: Iterable[Mapping[str, Any]],
    *,
    title: str = 'OCR-4 Bounding Boxes',
    confidence: tuple[float | None, float | None] = (None, None),
) -> None:
    """Plot one image with classified OCR block rectangles and confidence."""
    image = Image.open(image_path)
    block_list = list(blocks)
    figure, axis = plt.subplots(figsize=(12, 9))
    axis.imshow(image)
    seen_types = set()

    for block in block_list:
        x1, y1 = block.get('top_left_x'), block.get('top_left_y')
        x2, y2 = block.get('bottom_right_x'), block.get('bottom_right_y')
        if None in (x1, y1, x2, y2):
            continue
        block_type = block.get('type', 'unknown')
        color = BLOCK_COLORS.get(block_type, '#888888')
        axis.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.8, edgecolor=color, facecolor=color, alpha=0.16,
        ))
        axis.text(
            x1 + 3, y1 + 14, block_type,
            fontsize=7, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.95, edgecolor='none'),
        )
        seen_types.add(block_type)

    handles = [
        patches.Patch(facecolor=BLOCK_COLORS.get(block_type, '#888888'), label=block_type)
        for block_type in sorted(seen_types)
    ]
    if handles:
        axis.legend(handles=handles, loc='lower right', fontsize=8, title='Block classification')
    axis.set_title(f'{title} ({len(block_list)} blocks)', fontweight='bold')
    axis.axis('off')

    average, minimum = confidence
    average_text = f'{average:.3f}' if average is not None else 'N/A'
    minimum_text = f'{minimum:.3f}' if minimum is not None else 'N/A'
    figure.suptitle(title, fontsize=14, fontweight='bold')
    figure.text(
        0.5, 0.01,
        f'Page confidence: average {average_text} | minimum {minimum_text}',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F2F2F2', edgecolor='#BDBDBD'),
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.show()



def replace_images_in_markdown(markdown: str, images: Mapping[str, str]) -> str:
    """Replace image filenames in markdown with their base64 data."""
    for image_name, encoded_image in images.items():
        markdown = markdown.replace(
            f'![{image_name}]({image_name})',
            f'![{image_name}]({encoded_image})',
        )
    return markdown

def plot_page_bounding_boxes(
    page: Mapping[str, Any],
    *,
    title: str = 'OCR-4 Page Bounding Boxes',
    confidence: tuple[float | None, float | None] = (None, None),
) -> None:
    """Plot OCR blocks on a blank canvas using OCR page dimensions."""
    dimensions = page.get('dimensions') or {}
    width = dimensions.get('width', 800)
    height = dimensions.get('height', 1100)
    blocks = list(page.get('blocks') or [])
    figure, axis = plt.subplots(figsize=(8, 11))
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect('equal')
    axis.set_facecolor('#F8F9FA')
    seen_types = set()

    for block in blocks:
        x1, y1 = block.get('top_left_x'), block.get('top_left_y')
        x2, y2 = block.get('bottom_right_x'), block.get('bottom_right_y')
        if None in (x1, y1, x2, y2):
            continue
        block_type = block.get('type', 'unknown')
        color = BLOCK_COLORS.get(block_type, '#888888')
        axis.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.16,
        ))
        axis.text(
            x1 + 3, y1 + 14, block_type,
            fontsize=6, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor=color, alpha=0.95, edgecolor='none'),
        )
        seen_types.add(block_type)

    handles = [
        patches.Patch(facecolor=BLOCK_COLORS.get(block_type, '#888888'), label=block_type)
        for block_type in sorted(seen_types)
    ]
    if handles:
        axis.legend(handles=handles, loc='lower right', fontsize=8, title='Block classification')
    axis.set_title(f'{title} ({len(blocks)} blocks)', fontweight='bold')
    axis.set_xlabel('x (pixels)')
    axis.set_ylabel('y (pixels)')

    average, minimum = confidence
    average_text = f'{average:.3f}' if average is not None else 'N/A'
    minimum_text = f'{minimum:.3f}' if minimum is not None else 'N/A'
    figure.text(
        0.5, 0.01,
        f'Page confidence: average {average_text} | minimum {minimum_text}',
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F2F2F2', edgecolor='#BDBDBD'),
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.show()



def simple_combined_markdown(page: Mapping[str, Any]) -> str:
    """Inline OCR-extracted page images in its markdown."""
    images = {
        image['id']: image['image_base64']
        for image in page.get('images', [])
        if image.get('id') and image.get('image_base64')
    }
    return replace_images_in_markdown(page.get('markdown', ''), images)



def markdown_table_rows(table_markdown: str) -> list[list[str]]:
    """Parse markdown table rows while preserving empty cells."""
    rows = []
    for line in table_markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith('|'):
            continue
        cells = [cell.strip() for cell in stripped.strip('|').split('|')]
        if len(cells) <= 1:
            continue
        if all(cell.replace('-', '').replace(':', '').strip() == '' for cell in cells):
            continue
        rows.append(cells)
    return rows



def markdown_table_to_df(table_markdown: str) -> pd.DataFrame:
    """Convert markdown table text to a DataFrame without shifting blanks."""
    rows = markdown_table_rows(table_markdown)
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    body = [row[:len(header)] + [''] * max(0, len(header) - len(row)) for row in rows[1:]]
    return pd.DataFrame(body, columns=header).replace(r'^\s*$', pd.NA, regex=True)



def parse_money(value: Any) -> float:
    """Parse a currency/percentage-like value, returning NaN when unavailable."""
    if value is None or not str(value).strip():
        return float('nan')
    cleaned = str(value).replace('$', '').replace(',', '').replace('%', '').strip()
    if cleaned.lower() in {'na', 'n/a', 'nan', 'null'}:
        return float('nan')
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = f'-{cleaned[1:-1]}'
    try:
        return float(cleaned.strip('()'))
    except ValueError:
        return float('nan')



def extract_pdf_lines(pdf_path: str | Path) -> list[str]:
    """Extract source PDF lines for a local fallback analysis."""
    import fitz

    lines = []
    with fitz.open(pdf_path) as document:
        for page in document:
            lines.extend(page.get_text('text').splitlines())
    return lines


def extract_net_income_values_from_pdf(lines: Iterable[str]) -> dict[str, float | None]:
    """Extract the five-column net-income row while preserving blank cells."""
    line_list = list(lines)
    for index, line in enumerate(line_list):
        if 'net income' not in line.lower():
            continue
        values: list[float | None] = []
        for candidate_line in line_list[index + 1:index + 12]:
            candidate = candidate_line.strip()
            if candidate in {'', '[ _ ]', '—', '-', '_'}:
                values.append(None)
                continue
            numbers = re.findall(r'\$?\d[\d,]*(?:\.\d+)?', candidate)
            if numbers:
                values.extend(parse_money(number) for number in numbers)
        if len(values) >= 5:
            return dict(zip(
                ('Q1 Actual', 'Q2 Forecast', 'Q3 Actual', 'Q4 Projection', 'Full Year Total'),
                values[:5],
            ))
    return {}
