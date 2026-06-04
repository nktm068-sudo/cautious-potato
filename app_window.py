import os
import re
import threading
import customtkinter as ctk
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

print("🤖 Запускаю Союз AI...")
GGUF_REPO = "AlexWortega/qwen35-4b-soyuz-merged-gguf"
GGUF_FILE = "qwen35-4b-soyuz-merged.nomtp.Q4_K_M.gguf"
model_path = hf_hub_download(GGUF_REPO, GGUF_FILE)

llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, verbose=False)

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
SYSTEM_PROMPT = (
    "Ты — Союз AI, высокоинтеллектуальный, исключительно вежливый и учтивый ИИ-собеседник. "
    "Ты общаешься с пользователем строго на 'Вы', уважительно и грамотно, проявляя безупречные манеры. "
    "Ни в коем случае не называй пользователя по имени Никита. "
    "Ты поддерживаешь диалог на любые темы, даешь развернутые и интересные ответы. "
    "Перед тем как ответить, ты кратко размышляешь на русском языке внутри тегов <think> ... </think>, "
    "а затем выдаешь свой итоговый вежливый ответ."
)

history = [{"role": "system", "content": SYSTEM_PROMPT}]

def split_thinking(text):
    think_parts = _THINK.findall(text)
    answer = _THINK.sub("", text)
    if "<think>" in text and "</think>" not in text:
        i = text.index("<think>")
        think_parts.append(text[i + len("<think>"):])
        answer = text[:i]
    return answer.strip(), "\n".join(p.strip() for p in think_parts).strip()

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Союз AI — Панель Управления")
app.geometry("900x600")

app.grid_columnconfigure(0, weight=3)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)
app.grid_rowconfigure(1, weight=0)

chat_frame = ctk.CTkFrame(app)
chat_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

chat_display = ctk.CTkTextbox(chat_frame, font=("Arial", 14), state="disabled", wrap="word")
chat_display.pack(expand=True, fill="both", padx=5, pady=5)

input_frame = ctk.CTkFrame(app, fg_color="transparent")
input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

user_entry = ctk.CTkEntry(input_frame, placeholder_text="Напишите Ваше сообщение для Союз AI...", font=("Arial", 14))
user_entry.pack(side="left", expand=True, fill="x", padx=(0, 10))

think_frame = ctk.CTkFrame(app)
think_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

think_title = ctk.CTkLabel(think_frame, text="🧠 Ход мыслей Союз AI:", font=("Arial", 14, "bold"))
think_title.pack(padx=5, pady=5, anchor="w")

think_display = ctk.CTkTextbox(think_frame, font=("Arial", 12), state="disabled", wrap="word")
think_display.pack(expand=True, fill="both", padx=5, pady=5)

def append_to_box(textbox, text):
    textbox.configure(state="normal")
    textbox.insert("end", text)
    textbox.configure(state="disabled")
    textbox.see("end")

def clear_and_set_box(textbox, text):
    textbox.configure(state="normal")
    textbox.delete("1.0", "end")
    textbox.insert("1.0", text)
    textbox.configure(state="disabled")

def ai_stream_target(user_text):
    global history
    messages = history + [{"role": "user", "content": user_text}]
    
    append_to_box(chat_display, f"\n🤖 Союз AI: ")
    
    raw_output = ""
    for chunk in llm.create_chat_completion(messages=messages, stream=True):
        delta = chunk["choices"]["delta"].get("content", "")
        if delta:
            raw_output += delta
            answer, thinking = split_thinking(raw_output)
            
            if thinking:
                clear_and_set_box(think_display, thinking)
            
            current_displayed = chat_display.get("1.0", "end-1c").split("🤖 Союз AI: ")[-1]
            if answer and answer != current_displayed:
                new_tokens = answer[len(current_displayed):]
                append_to_box(chat_display, new_tokens)
                
    final_answer, _ = split_thinking(raw_output)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": final_answer})
    
    send_btn.configure(state="normal")

def send_message(event=None):
    user_text = user_entry.get().strip()
    if not user_text:
        return
        
    user_entry.delete(0, "end")
    append_to_box(chat_display, f"\n👤 Пользователь: {user_text}\n")
    clear_and_set_box(think_display, "Анализ запроса системой...")
    
    send_btn.configure(state="disabled")
    threading.Thread(target=ai_stream_target, args=(user_text,), daemon=True).start()

send_btn = ctk.CTkButton(input_frame, text="Отправить", command=send_message, font=("Arial", 14, "bold"))
send_btn.pack(side="right")

user_entry.bind("<Return>", send_message)

append_to_box(chat_display, "Здравствуйте! Система Союз AI успешно запущена и готова к высокотехнологичному диалогу. Какая тема Вас интересует?")

app.mainloop()
