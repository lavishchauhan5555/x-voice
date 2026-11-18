#!/usr/bin/env python3
"""Append verified IPA-v6 units found in multilingual Arrow datasets.

Input text must already be phonemized. Separate phones within a word with '|',
for example: k|a|t (spaces continue to represent word boundaries). This script
does not convert ordinary spelling to IPA.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

# Keep this extractor and normalization map aligned with
# x_voice.model.utils.str_to_list_ipa_v6. This avoids importing the full model
# package just to build a vocabulary.
TOKEN_EXTRACTOR = re.compile(r"([a-zA-Z\u00C0-\u02AF]+)|(.)")
IPA_NORMALIZATION_MAP = {
    "eɪɛ": "eɪ|ɛ", "zeɪɛ": "z|eɪ|ɛ", "teɪɛ": "t|eɪ|ɛ", "əeɪ": "ə|eɪ",
    "aɪɛ": "aɪ|ɛ", "taɪ": "t|aɪ", "jap": "ja|p", "jud": "ju|d",
    "jaɛ": "ja|ɛ", "ʃja": "ʃ|ja", "jat": "ja|t", "ɑja": "ɑ|ja",
    "əlɹ": "əl|ɹ", "əlf": "əl|f", "oʊw": "oʊ|w", "daʊ": "d|aʊ",
    "meɪ": "m|eɪ", "taʊ": "t|aʊ", "daɪ": "d|aɪ",
}
GARBAGE_CHARS = r"[&\u311c\u3067\u304b\u3059\u03c0%@]"


def str_to_list_ipa_v6(phonemized: str) -> list[str]:
    """Match the IPA-v6 chunk extraction rules used by X-Voice."""
    text = re.sub(GARBAGE_CHARS, "", phonemized)
    for key in sorted(IPA_NORMALIZATION_MAP, key=len, reverse=True):
        text = text.replace(key, IPA_NORMALIZATION_MAP[key])

    fields = []
    words = text.split(" ")
    for word_index, word in enumerate(words):
        if word:
            for phone_chunk in word.split("|"):
                if not phone_chunk:
                    continue
                for match in TOKEN_EXTRACTOR.finditer(phone_chunk):
                    alphabetic_chunk, symbol_chunk = match.groups()
                    fields.append(alphabetic_chunk or symbol_chunk)
        if word_index < len(words) - 1:
            fields.append(" ")
    return fields


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LANG_ID=/path/to/raw.arrow")
    language, path = value.split("=", 1)
    if not language or not path:
        raise argparse.ArgumentTypeError("Use LANG_ID=/path/to/raw.arrow")
    return language, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-vocab", required=True, type=Path)
    parser.add_argument("--output-vocab", required=True, type=Path)
    parser.add_argument("--input", required=True, action="append", type=parse_input,
                        metavar="LANG_ID=ARROW_PATH",
                        help="Repeat once per language/Arrow file; supports two source datasets per language.")
    parser.add_argument("--text-column", default="text",
                        help="Column containing phonemized IPA-v6 text (default: text, as in X-Voice prepared Arrow files).")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Only append a unit seen at least this often (default: 2).")
    parser.add_argument("--max-unit-length", type=int, default=8,
                        help="Refuse unusually long units that may be whole words (default: 8 characters).")
    parser.add_argument("--min-separated-ratio", type=float, default=0.80,
                        help="Require this fraction of nonempty rows to contain '|', guarding against raw words.")
    parser.add_argument("--report", type=Path, default=None,
                        help="CSV report path; defaults beside output vocab as *_vocab_report.csv.")
    args = parser.parse_args()

    try:
        from datasets import Dataset
    except ImportError as exc:
        parser.error("Install the project dependency first: pip install datasets")

    if not args.base_vocab.is_file():
        parser.error(f"Base vocabulary not found: {args.base_vocab}")
    if args.min_count < 1 or args.max_unit_length < 1 or not 0 <= args.min_separated_ratio <= 1:
        parser.error("--min-count and --max-unit-length must be >= 1; --min-separated-ratio must be in [0, 1]")

    old_lines = args.base_vocab.read_text(encoding="utf-8").splitlines()
    if len(old_lines) != len(set(old_lines)):
        parser.error("Base vocabulary contains duplicate lines; resolve duplicates before extending it.")
    existing = set(old_lines)

    counts_by_language: dict[str, Counter[str]] = defaultdict(Counter)
    for language, arrow_path in args.input:
        if not arrow_path.is_file():
            parser.error(f"Arrow file not found for {language}: {arrow_path}")
        dataset = Dataset.from_file(str(arrow_path))
        if args.text_column not in dataset.column_names:
            parser.error(
                f"Column {args.text_column!r} missing in {arrow_path}; "
                f"available columns: {dataset.column_names}"
            )

        nonempty = 0
        separated = 0
        for row in dataset:
            text = row[args.text_column]
            if text is None:
                continue
            if not isinstance(text, str):
                parser.error(f"Expected string values in {args.text_column!r}, got {type(text).__name__}")
            if not text.strip():
                continue
            nonempty += 1
            separated += "|" in text
            counts_by_language[language].update(
                unit for unit in str_to_list_ipa_v6(text) if unit != " "
            )

        ratio = separated / nonempty if nonempty else 0.0
        print(f"{language}: {len(dataset):,} rows, {nonempty:,} nonempty, "
              f"{ratio:.1%} rows contain phone separator '|'")
        if nonempty == 0 or ratio < args.min_separated_ratio:
            parser.error(
                f"{arrow_path} does not look sufficiently phonemized ({ratio:.1%} separated; "
                f"required {args.min_separated_ratio:.1%}). Select a phonemized IPA column."
            )

    total_counts: Counter[str] = Counter()
    for language_counts in counts_by_language.values():
        total_counts.update(language_counts)
    additions = sorted(
        token for token, count in total_counts.items()
        if token not in existing and count >= args.min_count and token.strip()
        and len(token) <= args.max_unit_length
    )

    args.output_vocab.parent.mkdir(parents=True, exist_ok=True)
    args.output_vocab.write_text("\n".join(old_lines + additions) + "\n", encoding="utf-8")
    report_path = args.report or args.output_vocab.with_name(args.output_vocab.stem + "_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8-sig", newline="") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(["unit", "total_count", "languages", "counts_by_language", "action"])
        for token, count in sorted(total_counts.items(), key=lambda item: (-item[1], item[0])):
            langs = [lang for lang, counter in counts_by_language.items() if counter[token]]
            if token in existing:
                action = "already_in_base_vocab"
            elif count < args.min_count:
                action = "below_min_count"
            elif len(token) > args.max_unit_length:
                action = "too_long_review_as_possible_word"
            else:
                action = "appended"
            per_language = ";".join(f"{lang}:{counts_by_language[lang][token]}" for lang in langs)
            writer.writerow([token, count, ";".join(langs), per_language, action])

    print(f"Base vocabulary: {len(old_lines):,} tokens")
    print(f"Appended units: {len(additions):,} (min count {args.min_count}, max length {args.max_unit_length})")
    print(f"Expanded vocabulary: {len(old_lines) + len(additions):,} tokens")
    print(f"Saved vocabulary: {args.output_vocab}")
    print(f"Saved audit report: {report_path}")
    print("Review the CSV and verify IPA units with fluent speakers before training.")


if __name__ == "__main__":
    main()
