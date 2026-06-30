from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytesseract

import config


def configure_tesseract() -> str:
    requested_lang = getattr(config, "OCR_LANG", os.getenv("TESSERACT_LANG", "eng"))
    explicit_cmd = getattr(config, "TESSERACT_CMD", os.getenv("TESSERACT_CMD", ""))
    explicit_data = getattr(config, "TESSDATA_PREFIX", os.getenv("TESSDATA_PREFIX", ""))

    candidate_cmds = [
        explicit_cmd,
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]

    tesseract_cmd = next((path for path in candidate_cmds if path and Path(path).exists()), "")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        print("[warn] Tesseract executable was not found. Set TESSERACT_CMD.")

    candidate_data_dirs = [
        explicit_data,
        str(Path(tesseract_cmd).parent / "tessdata") if tesseract_cmd else "",
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tessdata"),
    ]

    tessdata_dir = next((path for path in candidate_data_dirs if path and Path(path).exists()), "")
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir
    else:
        print("[warn] Tesseract tessdata directory was not found. Set TESSDATA_PREFIX.")
        return requested_lang

    available_langs = {traineddata.stem for traineddata in Path(tessdata_dir).glob("*.traineddata")}
    requested_langs = [lang for lang in requested_lang.split("+") if lang]
    missing_langs = [lang for lang in requested_langs if lang not in available_langs]
    usable_langs = [lang for lang in requested_langs if lang in available_langs]

    if missing_langs:
        print(f"[warn] Missing Tesseract language packs: {', '.join(missing_langs)}")
        if usable_langs:
            fallback_lang = "+".join(usable_langs)
            print(f"[warn] Falling back OCR language from '{requested_lang}' to '{fallback_lang}'.")
            return fallback_lang

    return requested_lang
