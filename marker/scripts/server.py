import multiprocessing as mp
import os
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Annotated
import click
import uvicorn

from concurrent.futures import ProcessPoolExecutor, Future
from fastapi.concurrency import run_in_threadpool

# ------------------------------------------------------------------------------
# FASTAPI APP SETUP (No GPU imports globally)
# ------------------------------------------------------------------------------
app = FastAPI()
UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

executor = None  # Will be set in main() block
pdftext_workers = 1  # Also set in main() block

# ------------------------------------------------------------------------------
# WORKER INITIALIZER
# ------------------------------------------------------------------------------
def worker_initializer(gpu_ids):
    """
    Runs in each child process *once*. We figure out which worker index we are,
    pick one GPU ID, then set `CUDA_VISIBLE_DEVICES` to that single ID.
    After that, the child sees only that one physical GPU.
    """
    worker_id = mp.current_process()._identity[0] - 1
    physical_gpu_id = gpu_ids[worker_id]  # e.g. "1" or "2" or "3"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = physical_gpu_id
    print(
        f"[Child pid={os.getpid()}] Assigned physical GPU: {physical_gpu_id} | "
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )

# ------------------------------------------------------------------------------
# SETUP MULTIPROCESSING (Called in __main__)
# ------------------------------------------------------------------------------
def setup_multiprocessing():
    """
    - Use 'spawn' start method to avoid CUDA init conflicts.
    - Parse parent's CUDA_VISIBLE_DEVICES (like "1,2,3") -> create one process per device.
    - Return an executor pinned to those devices.
    """
    mp.set_start_method("spawn", force=True)

    device_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_list = [s.strip() for s in device_str.split(",") if s.strip()]
    num_gpus = len(gpu_list)
    if num_gpus == 0:
        # Means user didn't set CUDA_VISIBLE_DEVICES or set it to empty
        # => no visible GPUs, fallback to CPU
        gpu_list = []
        num_gpus = 1
        print("No GPUs found in parent's CUDA_VISIBLE_DEVICES, using CPU-only worker.")
    else:
        print(f"Parent sees these GPU IDs in CUDA_VISIBLE_DEVICES={gpu_list}")

    pool_size = num_gpus
    print(f"Creating ProcessPoolExecutor with {pool_size} workers.")

    pdf_workers_env = int(os.environ.get("NUM_WORKERS", "1"))
    print(f"Setting 'pdftext_workers' to {pdf_workers_env}")

    if num_gpus > 0:
        # We'll create one child process per GPU ID
        # Each child sets its own CUDA_VISIBLE_DEVICES
        proc_pool = ProcessPoolExecutor(
            max_workers=pool_size,
            initializer=worker_initializer,
            initargs=(gpu_list,),
        )
    else:
        # If no GPUs, just do a single worker, CPU only
        proc_pool = ProcessPoolExecutor(max_workers=1)

    return proc_pool, pdf_workers_env

# ------------------------------------------------------------------------------
# WORKER FUNCTION: LAZY IMPORT GPU LIBS + MARKER
# ------------------------------------------------------------------------------
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
    This runs in a child process. We do a LAZY IMPORT of `torch` and `marker.*`
    so that CUDA is initialized only under the single-GPU environment.
    """
    # LAZY import everything that might init the GPU
    import torch
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.settings import settings
    import io
    import base64
    import traceback

    try:
        artifact_dict = create_model_dict()

        config_parser = ConfigParser({
            "filepath": filepath,
            "page_range": page_range,
            "languages": languages,
            "force_ocr": force_ocr,
            "paginate_output": paginate_output,
            "output_format": output_format,
        })
        config_dict = config_parser.generate_config_dict()

        # If there's exactly one visible GPU (because CUDA_VISIBLE_DEVICES=some_id),
        # torch.cuda.device_count() should be 1 => device="cuda:0"
        # Otherwise, fallback to "cpu".
        num_gpus_local = torch.cuda.device_count()
        if num_gpus_local > 0:
            device = "cuda"
        else:
            device = "cpu"

        config_dict["device"] = device
        config_dict["pdftext_workers"] = pdftext_workers
        print(f"Using GPU: {device}")
        print(f"Using PDFText workers: {pdftext_workers}")

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

# ------------------------------------------------------------------------------
# FASTAPI ENDPOINTS
# ------------------------------------------------------------------------------
class CommonParams(BaseModel):
    filepath: Annotated[Optional[str], Field(description="Path to the PDF file.")]
    page_range: Annotated[Optional[str], Field(...)] = None
    languages: Annotated[Optional[str], Field(...)] = None
    force_ocr: Annotated[bool, Field(...)] = False
    paginate_output: Annotated[bool, Field(...)] = False
    output_format: Annotated[str, Field(...)] = "markdown"


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
    # Avoid blocking event loop by using run_in_threadpool
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
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    future: Future = executor.submit(
        pdf_worker_function,
        temp_file_path,
        page_range,
        languages,
        force_ocr,
        paginate_output,
        output_format,
        pdftext_workers
    )
    result = await run_in_threadpool(future.result)
    os.remove(temp_file_path)
    return result


@app.get("/")
async def root():
    return HTMLResponse("<h1>Marker API - pinned GPUs</h1><p>Check /docs</p>")


# ------------------------------------------------------------------------------
# CLI ENTRY
# ------------------------------------------------------------------------------
@click.command()
@click.option("--port", type=int, default=6666, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    executor, pdftext_workers = setup_multiprocessing()
    server_cli()
