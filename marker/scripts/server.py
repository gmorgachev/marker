import os
import io
import base64
import traceback

import torch  # used just to count GPUs
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Annotated

from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

import click
import uvicorn

from concurrent.futures import ProcessPoolExecutor, Future
from itertools import cycle

################################################################################
# GLOBAL SETUP
################################################################################

# 1) Detect the number of visible GPUs.
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    print("No GPUs found. If you intended to use GPUs, check your CUDA setup.")
else:
    print(f"Found {num_gpus} visible GPU(s).")

# 2) Create a process pool of size = number of GPUs (or at least 1 if no GPU).
#    If no GPU is found, we can decide to use 1 or more CPU processes.
pool_size = num_gpus if num_gpus > 0 else 1
print(f"Creating ProcessPoolExecutor with {pool_size} worker(s).")
executor = ProcessPoolExecutor(max_workers=pool_size)

# 3) Round-robin cycle over GPU indices. If no GPU, this will just use [0].
gpu_indices = cycle(range(num_gpus))  # e.g. 0,1,0,1... if 2 GPUs

# 4) The number of PDF-text workers from env, default 1 if not set
pdftext_workers = int(os.environ.get("NUM_WORKERS", "1"))
print(f"Will set config_dict['pdftext_workers'] to {pdftext_workers}.")

# If you maintain any global models, they need to be re-initialized
# inside each process. So we won't load them here, but inside the worker function.


################################################################################
# THE "WORKER" FUNCTION FOR PDF CONVERSION
################################################################################

def pdf_worker_function(
    filepath: str,
    page_range: Optional[str],
    languages: Optional[str],
    force_ocr: bool,
    paginate_output: bool,
    output_format: str,
    gpu_id: int,
    pdftext_workers: int
):
    """
    Runs in a separate process. We'll do the actual marker -> PDFConverter steps here.
    We assume PdfConverter or underlying code can accept 'device' or something similar.
    """

    try:
        # Re-create the necessary data structures in each process
        # (if your library requires global initialization, do it here).
        # For example:
        artifact_dict = create_model_dict()
        
        # Build config dict using the same logic as your original `_convert_pdf`.
        # The "device" key is hypothetical — depends on PdfConverter or your OCR library.
        # Also set the pdftext_workers from the environment or a config var.
        config_parser = ConfigParser(
            {
                "filepath": filepath,
                "page_range": page_range,
                "languages": languages,
                "force_ocr": force_ocr,
                "paginate_output": paginate_output,
                "output_format": output_format,
            }
        )
        config_dict = config_parser.generate_config_dict()
        config_dict["device"] = f"cuda:{gpu_id}" if num_gpus > 0 else "cpu"
        config_dict["pdftext_workers"] = pdftext_workers

        converter = PdfConverter(
            config=config_dict,
            artifact_dict=artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        rendered = converter(filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata

        # Convert images to base64
        encoded_images = {}
        for k, v in images.items():
            byte_stream = io.BytesIO()
            v.save(byte_stream, format=settings.OUTPUT_IMAGE_FORMAT)
            encoded_images[k] = base64.b64encode(byte_stream.getvalue()).decode(
                settings.OUTPUT_ENCODING
            )

        return {
            "format": output_format,
            "output": text,
            "images": encoded_images,
            "metadata": metadata,
            "success": True,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


################################################################################
# FASTAPI APP
################################################################################

app = FastAPI()

UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API (Multi-GPU Example)</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="Path to the PDF file.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(description="Comma-separated page numbers or ranges. Example: 0,5-10,20"),
    ] = None
    languages: Annotated[
        Optional[str],
        Field(description="Comma-separated list of languages for OCR."),
    ] = None
    force_ocr: Annotated[
        bool,
        Field(description="Force OCR on all pages. Defaults to False."),
    ] = False
    paginate_output: Annotated[
        bool,
        Field(
            description=(
                "If True, separates each page in the output with a marker. Defaults False."
            )
        ),
    ] = False
    output_format: Annotated[
        str,
        Field(description="Output format: 'markdown', 'json', or 'html'. Default = markdown."),
    ] = "markdown"


@app.post("/marker")
async def convert_pdf(params: CommonParams):
    """
    If 'filepath' is provided, we read that local file directly.
    This endpoint runs the job in a separate process (one per GPU).
    """
    if not params.filepath:
        return {"success": False, "error": "No filepath given."}

    # Pick the next GPU from the round-robin
    gpu_id = next(gpu_indices) if num_gpus > 0 else 0

    # Schedule the task to the process pool
    future: Future = executor.submit(
        pdf_worker_function,
        params.filepath,
        params.page_range,
        params.languages,
        params.force_ocr,
        params.paginate_output,
        params.output_format,
        gpu_id,
        pdftext_workers,
    )

    # Wait for the result
    result = future.result()

    return result


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    languages: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(..., description="PDF file to convert", media_type="application/pdf"),
):
    """
    Same as above, but we handle an uploaded file. We'll save it to disk, schedule the job,
    then remove it.
    """
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        contents = await file.read()
        upload_file.write(contents)

    # Pick GPU in round-robin
    gpu_id = next(gpu_indices) if num_gpus > 0 else 0

    # Dispatch to pool
    future: Future = executor.submit(
        pdf_worker_function,
        upload_path,
        page_range,
        languages,
        force_ocr,
        paginate_output,
        output_format,
        gpu_id,
        pdftext_workers,
    )

    result = future.result()

    # Remove the uploaded file after processing
    os.remove(upload_path)

    return result


################################################################################
# CLI Entry Point (if desired)
################################################################################

@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    server_cli()
