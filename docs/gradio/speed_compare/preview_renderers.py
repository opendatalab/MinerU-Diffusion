import base64
import html
import io
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from speed_compare.config import resolve_layout_font_path


RENDERABLE_PROMPT_TYPES = {"table", "formula", "layout"}

OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
OTSL_TAGS = (OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)
OTSL_FIND_PATTERN = re.compile(r"(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>).*?(?=(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)|$)", flags=re.DOTALL)
LAYOUT_BOX_PATTERN = re.compile(
    r"<\|box_start\|>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*<\|box_end\|>.*?<\|ref_start\|>\s*(.*?)\s*<\|ref_end\|>",
    flags=re.DOTALL,
)


@dataclass
class TableCell:
    text: str
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    row_span: int = 1
    col_span: int = 1
    column_header: bool = False


def supports_render(prompt_type: str) -> bool:
    return prompt_type in RENDERABLE_PROMPT_TYPES


def render_output_html(prompt_type: str, text: str, image_path: str | Path | None = None) -> str:
    if prompt_type == "table":
        return _wrap_rendered_html(_render_table_html(text))
    if prompt_type == "layout":
        if not image_path:
            return _render_error("Layout render requires the source image.")
        return _wrap_rendered_html(_render_layout_html(image_path, text))
    if prompt_type == "formula":
        return _wrap_rendered_html(_render_formula_html(text))
    return _render_plain_text(text)


def _wrap_rendered_html(inner_html: str) -> str:
    return f'<div class="stream-output rendered-output">{inner_html}</div>'


def _render_plain_text(text: str) -> str:
    return f'<div class="stream-output"><pre class="rendered-pre">{html.escape(text)}</pre></div>'


def _render_error(message: str) -> str:
    return (
        '<div class="stream-output rendered-output">'
        f'<div class="rendered-error">{html.escape(message)}</div>'
        "</div>"
    )


def _render_table_html(otsl_text: str) -> str:
    try:
        table_html = convert_otsl_to_html(otsl_text)
    except Exception as exc:
        return _render_error(f"Table render failed: {exc}")

    if not table_html:
        return _render_error("No table structure detected.")
    return f'<div class="rendered-table-wrap">{table_html}</div>'


def otsl_extract_tokens_and_text(text: str) -> tuple[list[str], list[str]]:
    pattern = "(" + "|".join(re.escape(tag) for tag in OTSL_TAGS) + ")"
    tokens = re.findall(pattern, text)
    mixed = [chunk for chunk in re.split(pattern, text) if chunk.strip()]
    return tokens, mixed


def otsl_pad_to_rect(otsl_text: str) -> str:
    cleaned = otsl_text.strip()
    if OTSL_NL not in cleaned:
        return cleaned + OTSL_NL

    rows = []
    for line in cleaned.split(OTSL_NL):
        if not line:
            continue
        cells = OTSL_FIND_PATTERN.findall(line)
        if not cells:
            continue
        last_filled = 0
        for index, cell in enumerate(cells, start=1):
            if cell.startswith(OTSL_FCEL):
                last_filled = index
        rows.append((cells, len(cells), last_filled))

    if not rows:
        return OTSL_NL

    optimal_width = max(row[2] for row in rows)
    optimal_width = max(optimal_width, max(row[1] for row in rows))

    rendered_rows = []
    for cells, current_width, _ in rows:
        if current_width < optimal_width:
            cells = cells + [OTSL_ECEL] * (optimal_width - current_width)
        elif current_width > optimal_width:
            cells = cells[:optimal_width]
        rendered_rows.append("".join(cells))

    return OTSL_NL.join(rendered_rows) + OTSL_NL


def otsl_parse_texts(mixed_texts: list[str], tokens: list[str]) -> tuple[list[TableCell], list[list[str]]]:
    split_rows = []
    current_row = []
    for token in tokens:
        if token == OTSL_NL:
            if current_row:
                split_rows.append(current_row)
                current_row = []
            continue
        current_row.append(token)
    if current_row:
        split_rows.append(current_row)

    if split_rows:
        max_cols = max(len(row) for row in split_rows)
        for row in split_rows:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)

    rebuilt = []
    text_index = 0
    for row in split_rows:
        for token in row:
            rebuilt.append(token)
            if text_index < len(mixed_texts) and mixed_texts[text_index] == token:
                text_index += 1
                if text_index < len(mixed_texts) and mixed_texts[text_index] not in OTSL_TAGS:
                    rebuilt.append(mixed_texts[text_index])
                    text_index += 1
        rebuilt.append(OTSL_NL)
        if text_index < len(mixed_texts) and mixed_texts[text_index] == OTSL_NL:
            text_index += 1

    def count_right(row_idx: int, col_idx: int, valid_tokens: tuple[str, ...]) -> int:
        span = 0
        cursor = col_idx
        while cursor < len(split_rows[row_idx]) and split_rows[row_idx][cursor] in valid_tokens:
            span += 1
            cursor += 1
        return span

    def count_down(row_idx: int, col_idx: int, valid_tokens: tuple[str, ...]) -> int:
        span = 0
        cursor = row_idx
        while cursor < len(split_rows) and split_rows[cursor][col_idx] in valid_tokens:
            span += 1
            cursor += 1
        return span

    cells = []
    row_idx = 0
    col_idx = 0

    for index, item in enumerate(rebuilt):
        if item in (OTSL_FCEL, OTSL_ECEL):
            row_span = 1
            col_span = 1
            right_offset = 1
            cell_text = ""
            if item == OTSL_FCEL:
                cell_text = rebuilt[index + 1] if index + 1 < len(rebuilt) else ""
                right_offset = 2

            next_right = rebuilt[index + right_offset] if index + right_offset < len(rebuilt) else ""
            next_bottom = ""
            if row_idx + 1 < len(split_rows) and col_idx < len(split_rows[row_idx + 1]):
                next_bottom = split_rows[row_idx + 1][col_idx]

            if next_right in (OTSL_LCEL, OTSL_XCEL):
                col_span += count_right(row_idx, col_idx + 1, (OTSL_LCEL, OTSL_XCEL))
            if next_bottom in (OTSL_UCEL, OTSL_XCEL):
                row_span += count_down(row_idx + 1, col_idx, (OTSL_UCEL, OTSL_XCEL))

            cells.append(
                TableCell(
                    text=cell_text.strip(),
                    start_row=row_idx,
                    end_row=row_idx + row_span,
                    start_col=col_idx,
                    end_col=col_idx + col_span,
                    row_span=row_span,
                    col_span=col_span,
                )
            )

        if item in (OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL):
            col_idx += 1
        if item == OTSL_NL:
            row_idx += 1
            col_idx = 0

    return cells, split_rows


def export_to_html(table_cells: list[TableCell], num_rows: int, num_cols: int) -> str:
    if not table_cells:
        return ""

    grid = [
        [
            TableCell(
                text="",
                start_row=row,
                end_row=row + 1,
                start_col=col,
                end_col=col + 1,
            )
            for col in range(num_cols)
        ]
        for row in range(num_rows)
    ]
    for cell in table_cells:
        for row in range(min(cell.start_row, num_rows), min(cell.end_row, num_rows)):
            for col in range(min(cell.start_col, num_cols), min(cell.end_col, num_cols)):
                grid[row][col] = cell

    rows_html = []
    for row in range(num_rows):
        cells_html = []
        for col in range(num_cols):
            cell = grid[row][col]
            if cell.start_row != row or cell.start_col != col:
                continue
            attrs = []
            if cell.row_span > 1:
                attrs.append(f' rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                attrs.append(f' colspan="{cell.col_span}"')
            tag = "th" if cell.column_header else "td"
            cells_html.append(f"<{tag}{''.join(attrs)}>{html.escape(cell.text.strip())}</{tag}>")
        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")
    return f'<table class="rendered-table">{"".join(rows_html)}</table>'


def convert_otsl_to_html(otsl_text: str) -> str:
    normalized = otsl_pad_to_rect(otsl_text)
    tokens, mixed = otsl_extract_tokens_and_text(normalized)
    cells, rows = otsl_parse_texts(mixed, tokens)
    num_cols = max((len(row) for row in rows), default=0)
    return export_to_html(cells, len(rows), num_cols)


def _render_formula_html(text: str) -> str:
    formula = text.strip()
    if not formula:
        return _render_error("No formula content detected.")

    try:
        image_uri = _formula_to_data_uri(formula)
    except Exception as exc:
        return (
            '<div class="rendered-formula-fallback">'
            f'<div class="rendered-error">Formula render failed: {html.escape(str(exc))}</div>'
            f'<pre class="rendered-pre">{html.escape(formula)}</pre>'
            "</div>"
        )

    return (
        '<div class="rendered-formula-wrap">'
        f'<img class="rendered-image formula-image" src="{image_uri}" alt="Rendered formula" />'
        "</div>"
    )


def _formula_to_data_uri(formula: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    lines = _normalize_formula_lines(formula)
    wrapped_lines = [f"${line}$" for line in lines]

    width = max(6.0, min(14.0, max(len(line) for line in lines) * 0.18))
    height = max(1.8, min(10.0, 0.9 + len(lines) * 1.1))

    fig = plt.figure(figsize=(width, height), dpi=180, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    top = 0.88
    step = 0.78 / max(len(wrapped_lines), 1)
    for index, line in enumerate(wrapped_lines):
        ax.text(0.05, top - index * step, line, fontsize=24, va="top", ha="left", color="black")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    return _to_data_uri(buffer.getvalue(), "image/png")


def _normalize_formula_lines(formula: str) -> list[str]:
    content = formula.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.strip("`").strip()

    wrappers = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
    for left, right in wrappers:
        if content.startswith(left) and content.endswith(right):
            content = content[len(left) : len(content) - len(right)].strip()
            break

    lines = []
    for raw_line in content.replace("\r", "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for left, right in wrappers:
            if line.startswith(left) and line.endswith(right):
                line = line[len(left) : len(line) - len(right)].strip()
                break
        lines.append(line)

    return lines or [content]


def _render_layout_html(image_path: str | Path, prompt: str) -> str:
    try:
        image_uri = _layout_to_data_uri(image_path, prompt)
    except Exception as exc:
        return _render_error(f"Layout render failed: {exc}")
    return (
        '<div class="rendered-layout-wrap">'
        f'<img class="rendered-image layout-image" src="{image_uri}" alt="Rendered layout" />'
        "</div>"
    )


def _layout_to_data_uri(image_path: str | Path, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_layout_font(max(12, image.height // 100))

    for index, match in enumerate(LAYOUT_BOX_PATTERN.finditer(prompt)):
        x1, y1, x2, y2 = [int(value) for value in match.groups()[:4]]
        label = match.group(5).strip()
        rotate = "up"
        suffix = match.group(0)
        if "<|rotate_left|>" in suffix:
            rotate = "left"
        elif "<|rotate_right|>" in suffix:
            rotate = "right"
        elif "<|rotate_down|>" in suffix:
            rotate = "down"

        box = (
            int(x1 / 1000 * image.width),
            int(y1 / 1000 * image.height),
            int(x2 / 1000 * image.width),
            int(y2 / 1000 * image.height),
        )
        outline = _layout_outline_color(label)
        draw.rectangle(box, outline=outline, width=4)

        text = f"{index + 1}: {label} | {rotate}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_height = bottom - top
        text_width = right - left
        text_y = box[1] - text_height if box[1] > text_height else box[1] + text_height
        text_box = (box[0], text_y, box[0] + text_width + 6, text_y + text_height + 4)
        draw.rectangle(text_box, fill=outline)
        draw.text((box[0] + 3, text_y + 2), text, fill="white", font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _to_data_uri(buffer.getvalue(), "image/png")


def _load_layout_font(font_size: int):
    font_path = resolve_layout_font_path()
    if font_path and font_path.exists():
        return ImageFont.truetype(str(font_path), font_size)
    return ImageFont.load_default()


def _layout_outline_color(label: str) -> str:
    label = label.lower()
    if label == "text":
        return "crimson"
    if label == "table":
        return "dodgerblue"
    if label == "title":
        return "darkorchid"
    if label == "equation":
        return "forestgreen"
    if "caption" in label:
        return "orange"
    if "footnote" in label:
        return "magenta"
    if label == "image":
        return "yellowgreen"
    return "deepskyblue"


def _to_data_uri(payload: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
