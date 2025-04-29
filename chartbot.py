from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Chat history storage (optional)
chat_history_ids = None

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history_ids

    # Get user input from the request
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"response": "Please provide a valid message."})

    # Tokenize the user input and append it to the chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = (
        torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    )

    # Generate chatbot response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Controls response randomness
        top_p=0.9,  # Nucleus sampling
    )
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
