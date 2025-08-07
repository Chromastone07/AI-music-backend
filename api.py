
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pickle
import numpy as np
import generate
import train_model

app = FastAPI()

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("--- Loading pre-trained model and data... ---")
with open("notes_data.pkl", "rb") as f:
    data = pickle.load(f)

notes = data['notes']
n_vocab = data['n_vocab']


dummy_input = np.random.rand(1, 30, 1) 
model = train_model.create_network(dummy_input, n_vocab)
model.load_weights("final_model_weights.weights.h5")
print("✅ Model and data loaded successfully.")


@app.get("/generate")
async def generate_endpoint():
    print("--- Received request to generate music ---")
    
    prediction = generate.generate_notes(model, notes, n_vocab, data['pitchnames'])
    output_path = generate.create_midi(prediction, "demo_output.mid")
    return FileResponse(output_path, media_type='audio/midi', filename='ai_composition.mid')