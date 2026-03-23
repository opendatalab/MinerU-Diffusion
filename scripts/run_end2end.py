#!/usr/bin/env python3

import argparse
import html
import itertools
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from mineru_diffusion.utils.bbox import draw_bbox


STOP_STRINGS = ("<|endoftext|>", "<|im_end|>")
SYSTEM_PROMPT = "You are a helpful assistant."
LAYOUT_IMAGE_SIZE = (1036, 1036)
MIN_IMAGE_EDGE = 28
MAX_IMAGE_EDGE_RATIO = 50
PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}

TASK_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "[default]": "\nText Recognition:",
    "[layout]": "\nLayout Analysis:",
}

ANGLE_MAPPING = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}

LAYOUT_RE = re.compile(
    r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
)

OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
OTSL_TOKENS = {OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL}
OTSL_PATTERN = re.compile(
    "(" + "|".join(re.escape(token) for token in [OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + ")"
)


@dataclass
class ContentBlock:
    type: str
    bbox: list[float]
    angle: int | None = None
    content: str | None = None


@dataclass
class TableCell:
    text: str
    row_span: int
    col_span: int
    start_row: int
    start_col: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-step MinerU-Diffusion document parsing.")
    parser.add_argument("--model-path", default=None, help="Local MinerU-Diffusion HF model directory.")
    parser.add_argument("--image-path", required=True, help="Input page image path.")
    parser.add_argument("--output-path", default=None, help="Optional markdown output path.")
    parser.add_argument("--blocks-json-path", default=None, help="Optional parsed blocks JSON path.")
    parser.add_argument(
        "--save-layout-image",
        action="store_true",
        help="Save a copy of the input page with detected layout boxes overlaid.",
    )
    parser.add_argument("--layout-image-path", default=None, help="Optional output path for the layout visualization.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--layout-gen-length", type=int, default=2048)
    parser.add_argument("--content-gen-length", type=int, default=1024)
    parser.add_argument("--table-gen-length", type=int, default=2048)
    parser.add_argument("--formula-gen-length", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--remask-strategy", default="low_confidence_dynamic")
    parser.add_argument("--dynamic-threshold", type=float, default=0.95)
    parser.add_argument("--keep-paratext", action="store_true", help="Keep header/footer/page number blocks.")
    parser.add_argument("--verbose", action="store_true", help="Print per-block progress.")
    return parser.parse_args()


def resolve_default_model_path(repo_dir: Path) -> Path:
    runtime_paths = repo_dir / "docs" / "gradio" / "runtime_paths.json"
    if runtime_paths.exists():
        payload = json.loads(runtime_paths.read_text(encoding="utf-8"))
        model_path = payload.get("diffusion_model_path")
        if model_path:
            path = Path(model_path).expanduser()
            if path.exists():
                return path.resolve()
    default_path = repo_dir / "model"
    if default_path.exists():
        return default_path.resolve()
    raise FileNotFoundError("Unable to resolve model path. Pass --model-path explicitly.")


def resolve_layout_image_path(args: argparse.Namespace, image_path: Path) -> Path:
    if args.layout_image_path:
        return Path(args.layout_image_path).expanduser().resolve()
    return image_path.with_name(f"{image_path.stem}_layout.png")


def trim_response(text: str) -> str:
    for stop in STOP_STRINGS:
        text = text.split(stop, 1)[0]
    return text.strip()


def get_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize_by_need(image: Image.Image) -> Image.Image:
    edge_ratio = max(image.size) / min(image.size)
    if edge_ratio > MAX_IMAGE_EDGE_RATIO:
        width, height = image.size
        if width > height:
            new_w, new_h = width, math.ceil(width / MAX_IMAGE_EDGE_RATIO)
        else:
            new_w, new_h = math.ceil(height / MAX_IMAGE_EDGE_RATIO), height
        new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
        new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
        image = new_image
    if min(image.size) < MIN_IMAGE_EDGE:
        scale = MIN_IMAGE_EDGE / min(image.size)
        new_w, new_h = round(image.width * scale), round(image.height * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return image


def prepare_layout_image(image: Image.Image) -> Image.Image:
    return get_rgb_image(image).resize(LAYOUT_IMAGE_SIZE, Image.Resampling.BICUBIC)


def convert_bbox(raw_bbox: tuple[str, str, str, str]) -> list[float] | None:
    x1, y1, x2, y2 = map(int, raw_bbox)
    if any(coord < 0 or coord > 1000 for coord in (x1, y1, x2, y2)):
        return None
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [value / 1000.0 for value in (x1, y1, x2, y2)]


def parse_angle(tail: str) -> int | None:
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


def parse_layout_output(output: str) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for line in output.splitlines():
        match = LAYOUT_RE.match(line.strip())
        if not match:
            continue
        x1, y1, x2, y2, block_type, tail = match.groups()
        bbox = convert_bbox((x1, y1, x2, y2))
        if bbox is None:
            continue
        blocks.append(
            ContentBlock(
                type=block_type.lower(),
                bbox=bbox,
                angle=parse_angle(tail),
            )
        )
    return blocks


def crop_block_image(page_image: Image.Image, block: ContentBlock) -> Image.Image:
    image = get_rgb_image(page_image)
    width, height = image.size
    left = max(0, min(width - 1, math.floor(block.bbox[0] * width)))
    top = max(0, min(height - 1, math.floor(block.bbox[1] * height)))
    right = max(left + 1, min(width, math.ceil(block.bbox[2] * width)))
    bottom = max(top + 1, min(height, math.ceil(block.bbox[3] * height)))
    cropped = image.crop((left, top, right, bottom))
    if block.angle in (90, 180, 270):
        cropped = cropped.rotate(block.angle, expand=True)
    return resize_by_need(cropped)


def extract_otsl_tokens_and_text(raw_text: str) -> tuple[list[str], list[str]]:
    tokens = OTSL_PATTERN.findall(raw_text)
    text_parts = [part for part in OTSL_PATTERN.split(raw_text) if part and part.strip()]
    return tokens, text_parts


def count_span_right(rows: list[list[str]], row_idx: int, col_idx: int, span_tokens: set[str]) -> int:
    span = 0
    cursor = col_idx
    while cursor < len(rows[row_idx]) and rows[row_idx][cursor] in span_tokens:
        span += 1
        cursor += 1
    return span


def count_span_down(rows: list[list[str]], row_idx: int, col_idx: int, span_tokens: set[str]) -> int:
    span = 0
    cursor = row_idx
    while cursor < len(rows) and col_idx < len(rows[cursor]) and rows[cursor][col_idx] in span_tokens:
        span += 1
        cursor += 1
    return span


def convert_otsl_to_html(otsl_content: str) -> str:
    if otsl_content.startswith("<table") and otsl_content.endswith("</table>"):
        return otsl_content

    tokens, mixed_texts = extract_otsl_tokens_and_text(otsl_content)
    rows = [list(group) for is_nl, group in itertools.groupby(tokens, lambda item: item == OTSL_NL) if not is_nl]
    if not rows:
        return otsl_content.strip()

    max_cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append(OTSL_ECEL)

    normalized_texts: list[str] = []
    text_idx = 0
    for row in rows:
        for token in row:
            normalized_texts.append(token)
            if text_idx < len(mixed_texts) and mixed_texts[text_idx] == token:
                text_idx += 1
                if text_idx < len(mixed_texts) and mixed_texts[text_idx] not in OTSL_TOKENS:
                    normalized_texts.append(mixed_texts[text_idx])
                    text_idx += 1
        normalized_texts.append(OTSL_NL)
        if text_idx < len(mixed_texts) and mixed_texts[text_idx] == OTSL_NL:
            text_idx += 1

    cells: list[TableCell] = []
    row_idx = 0
    col_idx = 0
    for index, part in enumerate(normalized_texts):
        if part in (OTSL_FCEL, OTSL_ECEL):
            row_span = 1
            col_span = 1
            next_offset = 1
            cell_text = ""
            if index + 1 < len(normalized_texts) and normalized_texts[index + 1] not in OTSL_TOKENS:
                cell_text = normalized_texts[index + 1].strip()
                next_offset = 2
            next_right = normalized_texts[index + next_offset] if index + next_offset < len(normalized_texts) else ""
            next_down = rows[row_idx + 1][col_idx] if row_idx + 1 < len(rows) and col_idx < len(rows[row_idx + 1]) else ""
            if next_right in (OTSL_LCEL, OTSL_XCEL):
                col_span += count_span_right(rows, row_idx, col_idx + 1, {OTSL_LCEL, OTSL_XCEL})
            if next_down in (OTSL_UCEL, OTSL_XCEL):
                row_span += count_span_down(rows, row_idx + 1, col_idx, {OTSL_UCEL, OTSL_XCEL})
            cells.append(
                TableCell(
                    text=cell_text,
                    row_span=row_span,
                    col_span=col_span,
                    start_row=row_idx,
                    start_col=col_idx,
                )
            )
        if part in OTSL_TOKENS - {OTSL_NL}:
            col_idx += 1
        if part == OTSL_NL:
            row_idx += 1
            col_idx = 0

    html_parts = ["<table>"]
    for row in range(len(rows)):
        html_parts.append("<tr>")
        for col in range(max_cols):
            cell = next((item for item in cells if item.start_row == row and item.start_col == col), None)
            if cell is None:
                continue
            attrs = []
            if cell.row_span > 1:
                attrs.append(f' rowspan="{cell.row_span}"')
            if cell.col_span > 1:
                attrs.append(f' colspan="{cell.col_span}"')
            html_parts.append(f"<td{''.join(attrs)}>{html.escape(cell.text)}</td>")
        html_parts.append("</tr>")
    html_parts.append("</table>")
    return "".join(html_parts)


def wrap_equation(content: str) -> str:
    content = content.strip()
    if not content:
        return ""
    if not content.startswith("\\["):
        content = f"\\[\n{content}"
    if not content.endswith("\\]"):
        content = f"{content}\n\\]"
    return content


def render_block_content(block: ContentBlock) -> str:
    content = (block.content or "").strip()
    if not content:
        return ""
    if block.type == "table":
        return convert_otsl_to_html(content)
    if block.type == "equation":
        return wrap_equation(content)
    return content


def should_extract_block(block: ContentBlock) -> bool:
    return block.type not in {"image", "equation_block"}


def should_keep_block(block: ContentBlock, keep_paratext: bool) -> bool:
    if not block.content:
        return False
    if not keep_paratext and block.type in PARATEXT_TYPES:
        return False
    return True


class DiffusionRunner:
    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        max_length: int,
        block_size: int,
        temperature: float,
        remask_strategy: str,
        dynamic_threshold: float,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.torch_dtype = getattr(torch, dtype)
        self.max_length = max_length
        self.block_size = block_size
        self.temperature = temperature
        self.remask_strategy = remask_strategy
        self.dynamic_threshold = dynamic_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).eval().to(device)
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<|MASK|>")

    def infer(self, image: Image.Image, prompt: str, gen_length: int) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
        ]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        if isinstance(prompt_text, tuple):
            prompt_text = prompt_text[0]
        inputs = self.processor(
            images=[image],
            text=prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(torch.long).to(self.device)
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(torch.long).to(self.device)
        pixel_values = inputs["pixel_values"].to(self.torch_dtype).to(self.device)

        with torch.no_grad():
            response_ids, _, _ = self.model.generate(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                mask_token_id=self.mask_token_id,
                denoising_steps=self.block_size,
                gen_length=gen_length,
                block_length=self.block_size,
                temperature=self.temperature,
                remasking_strategy=self.remask_strategy,
                dynamic_threshold=self.dynamic_threshold,
                tokenizer=self.tokenizer,
                stopping_criteria=list(STOP_STRINGS),
            )

        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=False)
        return trim_response(response)


def run_end2end(args: argparse.Namespace) -> tuple[str, list[ContentBlock], dict]:
    repo_dir = REPO_DIR
    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else resolve_default_model_path(repo_dir)
    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    page_image = Image.open(image_path).convert("RGB")
    runner = DiffusionRunner(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        block_size=args.block_size,
        temperature=args.temperature,
        remask_strategy=args.remask_strategy,
        dynamic_threshold=args.dynamic_threshold,
    )

    metrics: dict[str, float | int] = {}
    start = time.perf_counter()
    layout_start = time.perf_counter()
    layout_output = runner.infer(prepare_layout_image(page_image), TASK_PROMPTS["[layout]"], args.layout_gen_length)
    metrics["layout_elapsed"] = time.perf_counter() - layout_start

    layout_image_path = None
    if args.save_layout_image:
        layout_image_path = resolve_layout_image_path(args, image_path)
        draw_bbox(str(image_path), layout_output, str(layout_image_path))

    blocks = parse_layout_output(layout_output)
    if args.verbose:
        print(f"[layout] detected {len(blocks)} blocks", file=sys.stderr)

    extract_start = time.perf_counter()
    for index, block in enumerate(blocks):
        if not should_extract_block(block):
            continue
        prompt = TASK_PROMPTS.get(block.type, TASK_PROMPTS["[default]"])
        gen_length = args.content_gen_length
        if block.type == "table":
            gen_length = args.table_gen_length
        elif block.type == "equation":
            gen_length = args.formula_gen_length
        block_image = crop_block_image(page_image, block)
        if args.verbose:
            print(f"[extract] {index:02d} {block.type} size={block_image.size}", file=sys.stderr)
        block.content = runner.infer(block_image, prompt, gen_length)
    metrics["extract_elapsed"] = time.perf_counter() - extract_start
    metrics["total_elapsed"] = time.perf_counter() - start
    metrics["num_blocks"] = len(blocks)

    rendered_parts = []
    for block in blocks:
        block.content = render_block_content(block)
        if should_keep_block(block, args.keep_paratext):
            rendered_parts.append(block.content)
    markdown = "\n\n".join(part for part in rendered_parts if part)
    if layout_image_path is not None:
        metrics["layout_image_path"] = str(layout_image_path)
    return markdown, blocks, metrics


def main() -> None:
    args = parse_args()
    markdown, blocks, metrics = run_end2end(args)

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"[saved] markdown -> {output_path}", file=sys.stderr)

    if args.blocks_json_path:
        blocks_json_path = Path(args.blocks_json_path).expanduser().resolve()
        blocks_json_path.parent.mkdir(parents=True, exist_ok=True)
        blocks_json_path.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "blocks": [asdict(block) for block in blocks],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[saved] blocks -> {blocks_json_path}", file=sys.stderr)

    print(f"[metrics] {json.dumps(metrics, ensure_ascii=False)}", file=sys.stderr)
    print(markdown)


if __name__ == "__main__":
    main()
