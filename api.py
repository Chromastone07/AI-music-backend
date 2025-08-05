# api.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import shutil
import os
import uuid
import numpy as np
import pickle

# Import our refactored scripts
import preprocess
import train_model
import generate

app = FastAPI()

origins = [
    "http://localhost:5173", # For local testing
    "https://your-project-name.netlify.app" # For the live website
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple in-memory dictionary to act as our "database" for job statuses
jobs = {}

def run_training_pipeline(job_id: str, file_data: list):
    """The background task for preprocessing and training."""
    try:
        jobs[job_id] = {"status": "processing", "message": "Saving and processing MIDI files..."}
        job_folder = f"data/{job_id}"
        os.makedirs(job_folder, exist_ok=True)
        
        for item in file_data:
            file_path = os.path.join(job_folder, item["filename"])
            with open(file_path, "wb") as buffer:
                buffer.write(item["content"])
                
        notes = preprocess.process_midi_folder(job_folder)
        if not notes:
            raise ValueError("Could not find any notes in the provided MIDI files.")

        network_input, network_output, n_vocab = preprocess.prepare_sequences(notes)
        
        # Save the notes and vocab size for the generation step
        with open(f"data/{job_id}/notes_data.pkl", "wb") as f:
            pickle.dump({'notes': notes, 'n_vocab': n_vocab}, f)

        jobs[job_id] = {"status": "training", "message": f"Training started on {len(notes)} notes..."}
        
        model = train_model.create_network(np.array(network_input), n_vocab)
        # We now pass the 'jobs' dictionary to the train function for the stop callback
        train_model.train(model, network_input, network_output, job_id, jobs)
        
        # After training finishes (or is stopped), mark it as complete
        jobs[job_id] = {"status": "complete", "message": "Training complete! Ready to generate."}
    except Exception as e:
        jobs[job_id] = {"status": "failed", "message": str(e)}


@app.post("/upload-and-train")
async def upload_and_train_endpoint(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "starting", "message": "Job accepted."}
    file_data = [{"filename": file.filename, "content": await file.read()} for file in files]
    background_tasks.add_task(run_training_pipeline, job_id, file_data)
    return {"message": "Upload successful. Training has started in the background.", "job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"message": "Job not found."})
    return job

@app.post("/stop/{job_id}")
async def stop_training(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found."})
    
    if job.get("status") == "training":
        jobs[job_id]["status"] = "stopping"
        jobs[job_id]["message"] = "Stop signal received. Finishing current epoch..."
        return {"message": "Stop signal sent to training job."}
    
    return JSONResponse(status_code=400, content={"message": "Job is not currently training."})

@app.get("/generate/{job_id}")
async def generate_endpoint(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "complete":
        return JSONResponse(status_code=400, content={"message": "Job is not complete or does not exist."})

    # Load the data needed for generation
    with open(f"data/{job_id}/notes_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Generate the music
    prediction = generate.generate_notes(job_id, data['notes'], data['n_vocab'])
    
    # Create the MIDI file
    output_path = generate.create_midi(prediction, f"data/{job_id}/output.mid")
    
    return FileResponse(output_path, media_type='audio/midi', filename='ai_composition.mid')