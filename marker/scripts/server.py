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

# We'll define a Pool globally (initialized in main())
pool = None  
pdftext_workers = 1

# ------------------------------------------------------------------------------
# 1) PIN EACH CHILD PROCESS TO A GPU (ROUND ROBIN)
# ------------------------------------------------------------------------------
def pin_to_one_gpu_round_robin(gpu_ids):
    """
    Each child process runs this once at startup.
    
    We do:
        worker_id = current_process()._identity[0] - 1
        assigned_gpu = gpu_ids[ worker_id % len(gpu_ids) ]
    
    So if the pool spawns more processes than len(gpu_ids), we still reuse 
    the same GPUs in a round-robin manner (rather than falling back to CPU).
    
    If no GPUs are available (gpu_ids=[]), we'll just do CPU-only.
    """
    worker_id = mp.current_process()._identity[0] - 1

    if not gpu_ids:
        # CPU-only scenario (no GPUs found in parent)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"[Child pid={os.getpid()}] No GPU => CPU only.")
        return

    gpu_index = worker_id % len(gpu_ids)  # round-robin
    chosen_gpu = gpu_ids[gpu_index]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen_gpu
    print(
        f"[Child pid={os.getpid()}] round-robin pinned to GPU {chosen_gpu} "
        f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
    )

# ------------------------------------------------------------------------------
# 2) CHILD WORKER FUNCTION (PROCESS ONE PDF, THEN EXIT)
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
    Because 'maxtasksperchild=1', each process runs this ONCE. 
    Then the process is killed, guaranteeing GPU memory is freed.
    """
    import io
    import base64
    import traceback

    # Lazy import so the parent never initializes CUDA or marker's GPU code
    import torch
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.settings import settings

    try:
        torch.cuda.empty_cache()  # Clear leftover GPU memory

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

        # If pinned to GPU, torch.cuda.device_count() == 1; else 0 => CPU
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
# 3) FASTAPI MODELS
# ------------------------------------------------------------------------------
class CommonParams(BaseModel):
    filepath: Annotated[Optional[str], Field(description="Local file path")] = None
    page_range: Optional[str] = None
    languages: Optional[str] = None
    force_ocr: bool = False
    paginate_output: bool = False
    output_format: str = "markdown"

# ------------------------------------------------------------------------------
# 4) FASTAPI ENDPOINTS
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return HTMLResponse(
        "<h1>Marker API - Round-Robin GPU Pinned, maxtasksperchild=1</h1>"
        "<p>Check /docs for usage.</p>"
    )

@app.post("/marker")
async def convert_pdf(params: CommonParams):
    """
    Single request for local file. Wait for job => return result.
    Up to 'len(gpu_list)' tasks can run in parallel. 
    Each request kills a child after finishing => frees GPU memory.
    """
    if not params.filepath:
        return {"success": False, "error": "No filepath given."}

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
    # Wait in background thread => not blocking the event loop
    result = await run_in_threadpool(async_result.get)
    return result

@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(None),
    languages: Optional[str] = Form(None),
    force_ocr: Optional[bool] = Form(False),
    paginate_output: Optional[bool] = Form(False),
    output_format: Optional[str] = Form("markdown"),
    file: UploadFile = File(..., description="PDF file", media_type="application/pdf"),
):
    """
    Upload + process. The worker is pinned to one GPU in round-robin.
    If 3 GPUs, up to 3 requests run in parallel; the 4th request waits.
    """
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        async_result = pool.apply_async(
            pdf_worker_function,
            (
                tmp_path,
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
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return result

# ------------------------------------------------------------------------------
# 5) CLI
# ------------------------------------------------------------------------------
@click.command()
@click.option("--port", type=int, default=6666, help="Port to run on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run on")
def server_cli(port: int, host: str):
    uvicorn.run(app, host=host, port=port)

# ------------------------------------------------------------------------------
# 6) MAIN
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Parse parent's CUDA_VISIBLE_DEVICES => e.g. "1,2,3"
    device_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_list = [g.strip() for g in device_str.split(",") if g.strip()]
    num_gpus = len(gpu_list)

    # If no GPU => 1 CPU process. If 3 GPUs => 3 processes, etc.
    if num_gpus == 0:
        print("No GPUs => CPU-only. pool_size=1")
        pool_size = 1
    else:
        print(f"GPU list: {gpu_list}")
        pool_size = num_gpus

    pdftext_workers = int(os.environ.get("NUM_WORKERS", "1"))
    print(f"pdftext_workers={pdftext_workers}")
    print(f"Creating Pool with {pool_size} process(es), maxtasksperchild=1")

    # We use round-robin GPU assignment, so new children won't revert to CPU
    pool = mp.Pool(
        processes=pool_size,
        initializer=pin_to_one_gpu_round_robin,
        initargs=(gpu_list,),
        maxtasksperchild=1
    )

    server_cli()
