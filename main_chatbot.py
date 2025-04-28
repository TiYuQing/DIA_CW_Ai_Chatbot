import torch
import re
import tkinter as tk
from tkinter import scrolledtext, ttk
from tkinter import font as tkfont
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from enhanced_product_db import EnhancedProductDB, SAMPLE_PRODUCTS
import json

class Phi3MiniChatbot:

    def __init__(self, db):
        self.db = db
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model configuration
        self.model_name = "microsoft/phi-3-mini-4k-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Quantization config for 4GB VRAM
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ) if torch.cuda.is_available() else None

        # Model loading with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=self.quantization_config
        )

        # Generation parameters
        self.generation_config = {
            "temperature": 0.5,
            "max_new_tokens": 128,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,  # Add this
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        self.sessions = {}  # Stores user conversation histories
        self.system_prompt = """[System]
            You are a laptop expert assistant. Help users find laptops by asking questions.
            Ask questions about user preference before recommending anything.
            If you do recommend anything, ONLY recommend laptops from the following database. Never mention other products.
            Keep response in 1 sentence.

            Use context:
            {context}

            Conversation history:
            {history}

            [User]
            {query}

            [Assistant]"""
    
    def _clean_response(self, text: str) -> str:
        """Improved response cleaning with error handling"""
        endings = ['.', '?', '\n']
        indices = []
        
        # Find all valid ending positions
        for e in endings:
            pos = text.find(e)
            if pos > 0:  # Ignore -1 and position 0
                indices.append(pos)
        
        # Handle cases with no valid endings
        if not indices:
            # Fallback: Take first complete sentence or first 100 chars
            sentences = re.split(r'(?<=[.!?]) +', text)
            if sentences:
                return sentences[0].strip()
            return text[:100].strip() if len(text) > 100 else text.strip()
        
        # Find earliest ending point
        first_end = min(indices)
        return text[:first_end + 1].strip()

    def get_db_context(self):
        """Get structured product info for LLM context"""
        return "\n".join([
            f"{p['name']} ({p['price']} RM): {p['specs']}"
            for p in SAMPLE_PRODUCTS  # Directly use sample data
        ])

    def generate_response(self, user_id: str, query: str) -> str:
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                'history': [],
                'preferences': {
                    'budget': None,
                    'use_case': None,
                    'performance': None,
                    'portability': None
                }
            }

        session = self.sessions[user_id]
        
        session['history'].append({"role": "user", "content": query})

        context = self.get_db_context()
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in session['history'][-4:]])

        prompt = self.system_prompt.format(
            context=context,
            history=history,
            query=query
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            return_attention_mask=True
        )

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **self.generation_config
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],  # Skip prompt
            skip_special_tokens=True
        )
        cleaned_response = self._clean_response(response)
        if not cleaned_response.strip():
            cleaned_response = "Could you please elaborate?"

        # Validate against database
        valid_products = [p['name'] for p in SAMPLE_PRODUCTS]
        if not any(p in cleaned_response for p in valid_products):
            cleaned_response = "Let me check our available options. What's your budget range?"

        self._update_preferences(session, query)

        session['history'].append({"role": "assistant", "content": cleaned_response})
        return cleaned_response.strip()

    def _update_preferences(self, session, query):
        budget_match = re.findall(r'(\d{3,5})', query)
        if budget_match:
            prices = list(map(int, budget_match))
            session['preferences']['budget'] = (min(prices), max(prices))
        
        for case in ['gaming', 'work', 'school']:
            if case in query.lower():
                session['preferences']['use_case'] = case
                
        if 'gpu' in query.lower() or 'graphics' in query.lower():
            session['preferences']['performance'] = 'gpu'
        elif 'cpu' in query.lower() or 'processor' in query.lower():
            session['preferences']['performance'] = 'cpu'
            
        if 'light' in query.lower() or 'portable' in query.lower():
            session['preferences']['portability'] = True
        elif 'heavy' in query.lower() or 'powerful' in query.lower():
            session['preferences']['portability'] = False

def initialize_database():
    db = EnhancedProductDB()
    if db.collection is None or db.collection.count() == 0:
        print("Initializing database with sample laptops...")
        db.initialize_db(SAMPLE_PRODUCTS)
    return db

class ChatbotUI:
    def __init__(self, chatbot):
        self.chatbot = chatbot

        self.window = tk.Tk()
        self.window.title("Smart Laptop Assistant")

        # Set the window size and background color
        self.window.geometry("600x500")
        self.window.config(bg="#f5f5f5")

        # Define custom fonts
        self.font_large = tkfont.Font(family="Helvetica", size=14)
        self.font_small = tkfont.Font(family="Helvetica", size=12)

        # Chat history area
        self.chat_history = scrolledtext.ScrolledText(self.window, wrap=tk.WORD, height=20, width=60, font=self.font_small, bg="#f0f0f0", fg="#333", bd=1, relief="solid")
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

        # Create a style object for customizing the Entry widget
        self.style = ttk.Style()
        self.style.configure("TEntry",
                             font=self.font_large,
                             padding=10)

        # User input field (Entry)
        self.entry = ttk.Entry(self.window, width=50, style="TEntry")
        self.entry.grid(row=1, column=0, padx=20, pady=10)

        # Send button
        self.send_button = ttk.Button(self.window, text="Send", command=self.send_message, style="SendButton.TButton")
        self.send_button.grid(row=1, column=1, padx=20, pady=10)

        # Apply style for the button (black background with white text)
        self.style.configure("SendButton.TButton", font=self.font_large, padding=10, width=12, relief="flat", background="black", foreground="white")
        self.style.map("SendButton.TButton", background=[("active", "#333333")])

        # Close window handler
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def send_message(self):
        user_input = self.entry.get().strip()
        if user_input.lower() in ("exit", "quit"):
            self.on_closing()
            return
        
        if not user_input:  # Prevent empty inputs
            return

        self.chat_history.insert(tk.END, f"You: {user_input}\n")
        self.chat_history.yview(tk.END)

        try:
            response = self.chatbot.generate_response("default_user", user_input)
        except Exception as e:
            response = "Apologies, I encountered an error. Please try again."

        self.chat_history.insert(tk.END, f"Assistant: {response}\n\n")
        self.chat_history.yview(tk.END)

        self.entry.delete(0, tk.END)

    def on_closing(self):
        self.window.destroy()

    def run(self):
        self.window.mainloop()

def chat():
    db = initialize_database()
    bot = Phi3MiniChatbot(db)
    ui = ChatbotUI(bot)
    ui.run()

if __name__ == "__main__":
    chat()
