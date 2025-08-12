from __future__ import annotations
import asyncio
import logging
import os
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

from .config import settings
from .rag_pipeline import RAG

logging.basicConfig(level=logging.INFO)
rag = RAG()

WELCOME = (
    "Привет! Я бот EORA. Задайте вопрос — я отвечу, опираясь на материалы сайта eora.ru и локальный архив.\n"
    "Ссылки на источники вставляю прямо в ответ."
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    q = (update.message.text or "").strip()
    if not q:
        return
    try:
        res = rag.answer(q)
        await update.message.reply_text(res["answer"], disable_web_page_preview=True)
        # Attach local files if any
        for fp in res.get("attachments", []):
            try:
                with open(fp, "rb") as f:
                    await update.message.reply_document(InputFile(f, filename=os.path.basename(fp)))
            except Exception:
                logging.exception("Failed to attach %s", fp)
    except FileNotFoundError:
        await update.message.reply_text(
            "Индекс не найден. Запустите индексацию: `python -m app.ingest`",
            parse_mode=None,
        )
    except Exception:
        logging.exception("Failure in /ask")
        await update.message.reply_text("Упс, что-то пошло не так. Попробуйте переформулировать вопрос.")


def main() -> None:
    application = ApplicationBuilder().token(settings.TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), ask))
    application.run_polling(close_loop=False)


if __name__ == "__main__":
    main()