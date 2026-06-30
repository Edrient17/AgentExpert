import os
import uuid

import matplotlib.pyplot as plt
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TableImageInput(BaseModel):
    markdown_string: str = Field(description="Markdown table text.")
    output_dir: str = Field(default="output/tables", description="Directory where the image file will be saved.")


@tool(args_schema=TableImageInput)
def create_table_image(markdown_string: str, output_dir: str = "output/tables") -> str:
    """
    Render a Markdown table as a PNG image and return the saved path.
    """
    print("[tool] Running Table Image tool...")
    try:
        plt.rc("font", family="NanumGothic")
        plt.rcParams["axes.unicode_minus"] = False

        if not markdown_string.strip():
            raise ValueError("The Markdown input is empty.")

        os.makedirs(output_dir, exist_ok=True)

        lines = [line.strip() for line in markdown_string.strip().split("\n")]
        if len(lines) < 3:
            raise ValueError("Invalid Markdown table format.")

        header = [h.strip() for h in lines[0].strip("|").split("|")]
        data = [[d.strip() for d in line.strip("|").split("|")] for line in lines[2:]]

        if not header or not data:
            raise ValueError("Could not parse table header or data.")

        fig, ax = plt.subplots(figsize=(len(header) * 1.5, len(data) * 0.5 + 1))
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(cellText=data, colLabels=header, loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.5, 1.5)
        table.auto_set_column_width(col=list(range(len(header))))

        file_name = f"{uuid.uuid4()}.png"
        file_path = os.path.join(output_dir, file_name)
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        print(f"[ok] Table image created: {file_path}")
        return file_path
    except Exception as e:
        print(f"[error] Table Image tool error: {e}")
        return f"Error: failed to generate table image. ({e})"
