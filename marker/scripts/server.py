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
from fastapi.concurrency import run_in_threadpool

# marker imports
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

app = FastAPI()
UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

################################################################################
# 1) Initializer to pin each process to a GPU
################################################################################

def init_gpu_visible_devices(gpu_ids):
    """
    This function runs in each child process at start-up.
    We pick the current worker ID to decide which GPU ID to use.
    Then we set CUDA_VISIBLE_DEVICES to that single GPU.
    """
    # The worker ID is typically the 1-based index in the process's name/identity.
    worker_id = mp.current_process()._identity[0] - 1
    # Grab the GPU ID from the list
    gpu_id = gpu_ids[worker_id]
    # Now set the environment variable so that only that GPU is visible
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Child pid={os.getpid()}] Assigned GPU {gpu_id}. "
          f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

################################################################################
# 2) Prepare the spawn method and create the process pool
################################################################################

def setup_multiprocessing():
    """
    Called inside 'if __name__ == "__main__":'
    - sets spawn start method
    - determines how many GPUs we have
    - creates a process pool with one process per GPU,
      each pinned to a distinct GPU.
    """
    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. If you intended to use GPUs, check CUDA setup.")
    else:
        print(f"Found {num_gpus} visible GPU(s).")

    # We'll have exactly one worker per GPU
    pool_size = num_gpus if num_gpus > 0 else 1
    print(f"Creating ProcessPoolExecutor with {pool_size} worker(s).")

    # For example, if num_gpus=2, we pass [0,1] so worker0 -> GPU0, worker1 -> GPU1
    gpu_ids = list(range(num_gpus))

    # The PDF workers from env
    pdftext_workers = int(os.environ.get("NUM_WORKERS", "1"))
    print(f"Will set config_dict['pdftext_workers'] to {pdftext_workers}.")

    # IMPORTANT: Provide initializer + initargs to pin each worker to a distinct GPU
    executor = ProcessPoolExecutor(
        max_workers=pool_size,
        initializer=init_gpu_visible_devices,
        initargs=(gpu_ids,)
    )

    return executor, pdftext_workers

executor = None
pdftext_workers = 1

################################################################################
# 3) Worker function that does the actual PDF => text conversion
################################################################################

def pdf_worker_function(
    filepath: str,
    page_range: Optional[str],
    languages: Optional[str],
    force_ocr: bool,
    paginate_output: bool,
    output_format: str,
    pdftext_workers: int
):
    """
    Because we set CUDA_VISIBLE_DEVICES in the initializer, from this process's
    perspective there is exactly ONE GPU visible, which is 'cuda:0' to PyTorch.
    """
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

        # On this child, the single visible GPU is now index 0
        # from the perspective of the process.
        if torch.cuda.device_count() > 0:
            device = "cuda"
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
# 4) FastAPI endpoints
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

    future: Future = executor.submit(
        pdf_worker_function,
        params.filepath,
        params.page_range,
        params.languages,
        params.force_ocr,
        params.paginate_output,
        params.output_format,
        pdftext_workers
    )
    # Use run_in_threadpool to avoid blocking the server's event loop
    result = await run_in_threadpool(future.result)
    return result

@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    languages: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(..., description="PDF file", media_type="application/pdf"),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as out_file:
        out_file.write(await file.read())

    future: Future = executor.submit(
        pdf_worker_function,
        upload_path,
        page_range,
        languages,
        force_ocr,
        paginate_output,
        output_format,
        pdftext_workers
    )
    result = await run_in_threadpool(future.result)
    os.remove(upload_path)
    return result

@app.get("/")
async def root():
    return HTMLResponse("<h1>Marker API (Per-Process CUDA_VISIBLE_DEVICES)</h1><p>Check /docs</p>")

################################################################################
# 5) CLI Entry
################################################################################

@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    executor, pdftext_workers = setup_multiprocessing()
    server_cli()
