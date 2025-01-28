import multiprocessing as mp
import os
import io
import base64
import traceback
import torch
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Annotated
import click
import uvicorn

from concurrent.futures import ProcessPoolExecutor, Future
from itertools import cycle

# marker imports
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings
from fastapi.concurrency import run_in_threadpool

app = FastAPI()
UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

################################################################################
# Move GPU-using logic into a guarded function
################################################################################

def setup_multiprocessing():
    """
    Called inside 'if __name__ == "__main__":' to set spawn start method
    and create the process pool, among other global variables.
    Returns a tuple (executor, gpu_indices, pdftext_workers).
    """
    mp.set_start_method("spawn", force=True)  # <--- KEY LINE

    # Now we can safely do torch-related stuff
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. If you intended to use GPUs, check CUDA setup.")
    else:
        print(f"Found {num_gpus} visible GPU(s).")

    # Create process pool
    pool_size = num_gpus if num_gpus > 0 else 1
    print(f"Creating ProcessPoolExecutor with {pool_size} worker(s).")
    executor = ProcessPoolExecutor(max_workers=pool_size)

    # Round-robin cycle over GPU indices
    gpu_indices = cycle(range(num_gpus))  # e.g. 0,1,0,1... if 2 GPUs

    # Number of PDF-text workers from env
    pdftext_workers = int(os.environ.get("NUM_WORKERS", "1"))
    print(f"Will set config_dict['pdftext_workers'] to {pdftext_workers}.")

    return executor, gpu_indices, pdftext_workers

# We'll define placeholders to be filled in at runtime.
executor = None
gpu_indices = None
pdftext_workers = 1

################################################################################
# Worker function
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
    # Child process logic
    try:
        artifact_dict = create_model_dict()

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

        # Use the GPU in child
        num_gpus_local = torch.cuda.device_count()
        if num_gpus_local > 0:
            device = f"cuda:{gpu_id}" 
        else:
            device = "cpu"

        config_dict["device"] = device
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
        return {"success": False, "error": str(e)}

################################################################################
# FastAPI endpoints
################################################################################

class CommonParams(BaseModel):
    filepath: Annotated[Optional[str], Field(description="Path to the PDF file.")]
    page_range: Annotated[Optional[str], Field(description="Comma-separated pages/ranges")] = None
    languages: Annotated[Optional[str], Field(description="Comma-separated OCR languages.")] = None
    force_ocr: Annotated[bool, Field(description="Force OCR on all pages.")] = False
    paginate_output: Annotated[bool, Field(description="Separate each page in output.")] = False
    output_format: Annotated[str, Field(description="Output format.")] = "markdown"

@app.post("/marker")
async def convert_pdf(params: CommonParams):
    if not params.filepath:
        return {"success": False, "error": "No filepath given."}

    gpu_id = next(gpu_indices) if torch.cuda.device_count() > 0 else 0

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
    return future.result()

@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    languages: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(..., description="PDF file", media_type="application/pdf"),
):
    print(f"Uploading file: {file.filename}")
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as out_file:
        out_file.write(await file.read())

    gpu_id = next(gpu_indices) if torch.cuda.device_count() > 0 else 0
    print(f"Using GPU: {gpu_id}")
    print(f"Using PDFText workers: {pdftext_workers}")

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
    result = await run_in_threadpool(future.result)

    os.remove(upload_path)
    return result

@app.get("/")
async def root():
    return HTMLResponse("<h1>Marker API with Spawn</h1><p>Check /docs</p>")

################################################################################
# CLI Entry
################################################################################

@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # 1) Initialize spawn start method and multiprocess resources
    executor, gpu_indices, pdftext_workers = setup_multiprocessing()

    # 2) Run the CLI
    server_cli()
