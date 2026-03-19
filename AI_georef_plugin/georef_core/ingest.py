from __future__ import annotations

from pathlib import Path

from .models import IngestResult, PipelineContext
from .runtime import load_auto_georeference


def ingest_plan(context: PipelineContext) -> IngestResult:
    ag = load_auto_georeference()

    if context.is_pdf:
        metadata = ag.pdf_metadata(context.input_path)
        pdf_text = ag.extract_pdf_text(context.input_path) or ""
        rendered_img = ag.render_pdf_to_image(context.input_path)
        rendered_tif = context.artifact_dir / f"{context.input_path.stem}_rendered.tif"
        rendered_img.save(str(rendered_tif), format="TIFF", compression="lzw", dpi=(ag.PDF_RENDER_DPI, ag.PDF_RENDER_DPI))
        return IngestResult(
            source_path=context.input_path,
            working_path=rendered_tif,
            is_pdf=True,
            metadata=metadata,
            pdf_text=pdf_text,
            rendered_from_pdf=True,
        )

    metadata = ag.read_tiff_metadata(context.input_path)
    return IngestResult(
        source_path=context.input_path,
        working_path=context.input_path,
        is_pdf=False,
        metadata=metadata,
    )
