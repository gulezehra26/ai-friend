# AI Friend – Your Everyday Companion

AI Friend is a friendly AI chatbot designed for natural, everyday conversations. It can chat, share general knowledge, help with ideas, and provide a comfortable space to express thoughts.

## Features

- Natural conversations in English and Pakistani Roman Urdu.
- Adaptive language and conversational style.
- Gemini API as the primary AI provider.
- Ollama as a local backup provider.
- Current-session conversation memory.
- Basic API rate limiting.
- Privacy notice and input validation.
- Responsive, pastel-themed chat interface.

## Technologies Used

- Python
- FastAPI
- Google Gemini API
- Ollama
- HTML
- CSS
- JavaScript
- SlowAPI

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/gulezehra26/ai-friend.git
   ```

2. Open the project folder:

   ```bash
   cd ai-friend
   ```

3. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Install dependencies:

   ```bash
   pip install fastapi uvicorn ollama google-genai python-dotenv slowapi
   ```

5. Create a `.env` file in the project folder and add your Gemini API key:

   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

6. Install Ollama and download the backup model:

   ```bash
   ollama pull qwen2.5:1.5b
   ```

7. Start the backend:

   ```bash
   uvicorn main:app --reload
   ```

8. Open `index.html` in your browser to use the chatbot.

##  Privacy & Security

- API keys are stored in environment variables and should never be committed to GitHub.
- Conversation history is maintained temporarily in server memory.
- The application does not currently use permanent database storage.
- Messages are processed by the Gemini API when generating responses.
- Users should avoid sharing sensitive personal information.

##  Project Status

This is a portfolio demonstration project. It is still being developed and has not yet been prepared for production use.

##  Author

Developed as a personal AI chatbot project to explore conversational AI, backend development, and frontend integration.