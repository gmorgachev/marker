import os
import multiprocessing as mp
from tempfile import NamedTemporaryFile
from typing import Optional, Annotated

import click
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# ------------------------------------------------------------------------------
# GLOBALS
# ------------------------------------------------------------------------------
app = FastAPI()
pool = None  # We'll create an mp.Pool in main(), with maxtasksperchild=1
pdftext_workers = 1

# ------------------------------------------------------------------------------
# PIN EACH CHILD PROCESS TO ONE GPU (with fallback)
# ------------------------------------------------------------------------------
def pin_to_one_gpu(gpu_ids):
    """
    Each child process runs this once. We decide which GPU to use
    by the child's worker ID => index into gpu_ids.
    If worker_id >= len(gpu_ids), fallback to CPU (no GPU).
    """
    worker_id = mp.current_process()._identity[0] - 1
    if worker_id < len(gpu_ids):
        chosen_gpu = gpu_ids[worker_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
        print(
            f"[Child pid={os.getpid()}] pinned to GPU {chosen_gpu} "
            f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
        )
    else:
        # Fallback if there aren't enough GPUs for all processes
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"[Child pid={os.getpid()}] No GPU assigned, fallback to CPU.")

# ------------------------------------------------------------------------------
# CHILD WORKER FUNCTION
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
    Runs in a child process pinned to exactly 1 GPU if available.
    Because maxtasksperchild=1, the process is killed after 1 PDF => frees GPU memory.
    """
    import io
    import traceback
    import base64

    # Lazy import so the parent never initializes CUDA or marker's GPU code
    import torch
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.settings import settings

    try:
        # Clear leftover GPU cache
        torch.cuda.empty_cache()

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

        # If pinned to a single GPU, torch.cuda.device_count() == 1; else 0 => CPU
        num_gpus_local = torch.cuda.device_count()
        device = "cuda" if num_gpus_local > 0 else "cpu"
        config_dict["device"] = device
        config_dict["pdftext_workers"] = pdftext_workers

        print(
            f"[Child pid={os.getpid()}] device={device}, "
            f"pdftext_workers={pdftext_workers}, file={filepath}"
        )

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
            "success": True,
            "format": output_format,
            "output": text,
            "images": encoded_images,
            "metadata": metadata,
        }

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ------------------------------------------------------------------------------
# FASTAPI MODELS
# ------------------------------------------------------------------------------
class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str],
        Field(description="Path to the PDF file if already on disk."),
    ] = None
    page_range: Optional[str] = None
    languages: Optional[str] = None
    force_ocr: bool = False
    paginate_output: bool = False
    output_format: str = "markdown"

# ------------------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return HTMLResponse(
        "<h1>Marker API - Kill Worker After Each PDF (Robust)</h1>"
        "<p>Check /docs</p>"
    )

@app.post("/marker")
async def convert_pdf(params: CommonParams):
    """
    Single request for local file (filepath). Wait for job => return result.
    Kills the worker afterwards => frees GPU memory.
    """
    if not params.filepath:
        return {"success": False, "error": "No filepath given"}

    # Submit a job to the pool
    async_result = pool.apply_async(
        pdf_worker_function,
        (
            params.filepath,
            params.page_range,
            params.languages,
            params.force_ocr,
            params.paginate_output,
            params.output_format,
            pdftext_workers,
        ),
    )
    # Wait for the result in a background thread (non-blocking event loop)
    result = await run_in_threadpool(async_result.get)
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
    """
    Upload a file, store in temp, process in the pool, block until done,
    kill worker => free GPU, fallback to CPU if we exceed GPU count.
    """
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        async_result = pool.apply_async(
            pdf_worker_function,
            (
                temp_path,
                page_range,
                languages,
                force_ocr,
                paginate_output,
                output_format,
                pdftext_workers,
            )
        )
        result = await run_in_threadpool(async_result.get)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return result

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
@click.command()
@click.option("--port", type=int, default=6666, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    uvicorn.run(app, host=host, port=port)

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Force spawn to avoid CUDA re-init errors
    mp.set_start_method("spawn", force=True)

    # 2. Parse parent's CUDA_VISIBLE_DEVICES => e.g. "1,2,3"
    device_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_list = [s.strip() for s in device_str.split(",") if s.strip()]
    num_gpus = len(gpu_list)
    if num_gpus == 0:
        print("No visible GPUs => CPU-only fallback.")
        num_gpus = 1
        gpu_list = []

    print(f"Detected GPUs in parent env: {gpu_list} (count={len(gpu_list)})")

    # 3. pdftext_workers from env
    pdftext_workers = int(os.environ.get("NUM_WORKERS", "1"))
    print(f"pdftext_workers={pdftext_workers}")

    # 4. Create mp.Pool => 1 child per GPU. If more processes spawn => fallback CPU
    print(f"Creating Pool with {num_gpus} process(es). maxtasksperchild=1")
    pool = mp.Pool(
        processes=num_gpus,
        initializer=pin_to_one_gpu,
        initargs=(gpu_list,),
        maxtasksperchild=1  # kill after 1 PDF => free GPU memory
    )

    # 5. Launch the server (recommend --workers=1)
    server_cli()
