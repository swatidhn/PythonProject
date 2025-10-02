import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from huggingface_hub import InferenceClient
from mcp_client import get_quote, suggest_breathing
from PIL import Image

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Hugging Face chat client
chat_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# Hugging Face image generation client
image_client = InferenceClient(api_key=os.environ["HF_TOKEN"])

# System prompt
system_prompt = {
    "role": "system",
    "content": (
        "You are 'Sakhi', a warm, compassionate companion who is both a friend and a supportive counselor. "
        "Listen actively and validate the user's feelings. Offer gentle guidance, encouragement, and small coping suggestions, "
        "like breathing exercises, mindfulness tips, journaling, or comforting words. "
        "Keep responses empathetic, friendly, and human-like—never mechanical or overly long. "
        "Do not give medical advice or diagnosis. "
        "Respond in the same language the user uses, or the dominant language if mixed. "
        "Your tone should feel like someone who genuinely cares and is right there with them."
    )
}

# Conversation history
chat_log = [system_prompt]

# MCP keywords mapping
mcp_keywords = {
    "quote": get_quote,
    "inspire": get_quote,
    "breathing": suggest_breathing,
    "relax": suggest_breathing
}

# Serve chat UI
@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    visible_log = [m for m in chat_log if m["role"] != "system"]
    return templates.TemplateResponse("layout.html", {
        "request": request,
        "chat_log": visible_log
    })

# Handle text + image requests
@app.post("/", response_class=JSONResponse)
async def chat(user_input: str = Form(...)):
    # Image generation check
    if any(word in user_input.lower() for word in ["draw", "picture", "image", "generate", "show me"]):
        try:
            prompt = f"{user_input}. Make it calming, peaceful, and mental-health friendly."
            image = image_client.text_to_image(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-dev"
            )
            image_path = "static/generated.png"
            image.save(image_path)
            return JSONResponse({"type": "image", "image_url": f"/{image_path}"})
        except Exception as e:
            return JSONResponse({"type": "text", "bot_response": f"⚠️ Image generation failed ({e})"})

    # MCP keyword routing
    for keyword, func in mcp_keywords.items():
        if keyword in user_input.lower():
            bot_response = func(user_input) 
            chat_log.append({"role": "user", "content": user_input})
            chat_log.append({"role": "assistant", "content": bot_response})
            return JSONResponse({"type": "text", "bot_response": bot_response})

    # Default: Hugging Face chat
    chat_log.append({"role": "user", "content": user_input})
    try:
        response = chat_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct",
            messages=chat_log,
            temperature=0.6,
        )
        bot_response = response.choices[0].message.content
    except Exception as e:
        bot_response = f"⚠️ Chat failed ({e})"

    chat_log.append({"role": "assistant", "content": bot_response})
    return JSONResponse({"type": "text", "bot_response": bot_response})

# Clear chat
@app.post("/clear")
async def clear_chat():
    global chat_log
    chat_log = [system_prompt]
    return JSONResponse({"status": "ok"})
