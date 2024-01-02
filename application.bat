@echo off
call "llm_env\Scripts\activate"
chainlit run dev_scripts\chatbot.py %*
pause