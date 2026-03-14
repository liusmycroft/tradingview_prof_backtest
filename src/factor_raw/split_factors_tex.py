#!/usr/bin/env python3
"""Split a LaTeX factor file into multiple chunked TeX files by \\section.

Default behavior:
- Input:  factors.tex
- Output directory: factors_split
- Chunk size: 10 sections per file
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


SECTION_PATTERN = re.compile(r"^\s*\\section\{", re.MULTILINE)


def split_content_by_sections(content: str) -> tuple[str, list[str], str]:
    """Return (preamble, sections, tail).

    - preamble: text before the first \\section
    - sections: each section block starts at \\section and ends before next \\section
    - tail: text after the last section block (usually \\end{document})
    """
    matches = list(SECTION_PATTERN.finditer(content))
    if not matches:
        raise ValueError("未找到任何 \\section，无法按因子拆分。")

    preamble = content[: matches[0].start()]
    sections: list[str] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        sections.append(content[start:end])

    # Extract tail from the last section: if \end{document} exists, keep it as tail.
    last_section = sections[-1]
    end_doc_idx = last_section.rfind(r"\end{document}")
    if end_doc_idx != -1:
        tail = last_section[end_doc_idx:]
        sections[-1] = last_section[:end_doc_idx]
    else:
        tail = "\n"

    return preamble, sections, tail


def write_split_files(
    preamble: str,
    sections: list[str],
    tail: str,
    output_dir: Path,
    chunk_size: int,
    prefix: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(sections)
    file_count = math.ceil(total / chunk_size)
    width = max(2, len(str(file_count)))

    for file_idx in range(file_count):
        start = file_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk_sections = sections[start:end]

        filename = f"{prefix}_{file_idx + 1:0{width}d}.tex"
        output_path = output_dir / filename

        # Ensure each generated file is a standalone compilable .tex file.
        output_text = preamble + "".join(chunk_sections) + tail
        output_path.write_text(output_text, encoding="utf-8")

    return file_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按每 N 个 \\section 拆分 factors.tex 为多个 .tex 文件。"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="factors.tex",
        help="输入的 .tex 文件路径（默认：factors.tex）",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="factors_split",
        help="输出目录（默认：factors_split）",
    )
    parser.add_argument(
        "-n",
        "--chunk-size",
        type=int,
        default=10,
        help="每个输出文件包含的 \\section 数量（默认：10）",
    )
    parser.add_argument(
        "--prefix",
        default="factors_part",
        help="输出文件名前缀（默认：factors_part）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.chunk_size <= 0:
        raise ValueError("chunk-size 必须是正整数。")
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    content = input_path.read_text(encoding="utf-8")
    preamble, sections, tail = split_content_by_sections(content)
    file_count = write_split_files(
        preamble=preamble,
        sections=sections,
        tail=tail,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        prefix=args.prefix,
    )

    print(
        f"拆分完成：共 {len(sections)} 个因子（\\section），"
        f"每 {args.chunk_size} 个一组，生成 {file_count} 个文件。"
    )
    print(f"输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
