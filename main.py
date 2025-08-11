from fastapi import FastAPI, HTTPException, Header
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pydantic import BaseModel
import os

app = FastAPI()

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

VALID_API_KEYS = [os.getenv("API_KEY", "default_key_here")]

class TextInput(BaseModel):
    text: str


# Dependency to check API key
def api_key_header(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

@app.post("/predict/")
async def predict(text_input: TextInput, x_api_key: str = Header(...)):
    api_key_header(x_api_key)
    inputs = tokenizer(text_input.text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"])
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
