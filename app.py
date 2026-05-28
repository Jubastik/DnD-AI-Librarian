import streamlit as st
import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

from src.core.config import settings
from src.vector.embedders import DenseMultilingualE5SmallSparseBm25Embeder
from src.vector.qdrant.qdrant_engine import QdrantEngine
from src.llm.providers import GroqLLM, ModelRateLimitError
from src.rag.rag_service import DndRagService

load_dotenv()

st.set_page_config(
    page_title="D&D RAG Master",
    page_icon="🐉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .stChatInput {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


def load_models_config():
    try:
        with open("models_config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Файл models.yaml не найден!")
        st.stop()


config = load_models_config()
MODELS_MAP = {m["name"]: m for m in config["models"]}


def create_llm_instance(model_config: Dict[str, Any]):
    provider = model_config.get("provider", "").lower()
    model_id = model_config.get("model_id")
    params = model_config.get("params", {})

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        return GroqLLM(model_name=model_id, api_key=api_key, **params)
    else:
        raise ValueError(f"Неизвестный провайдер: {provider}")


@st.cache_resource(show_spinner="Загрузка эмбедера и подключение к базе знаний...")
def get_engine():
    """
    Создает и собирает все компоненты системы в единый сервис.
    """
    try:
        print("🛠 Инициализация Embedder...")
        embedder = DenseMultilingualE5SmallSparseBm25Embeder()

        # print("⚖️ Инициализация Reranker...")
        # reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")

        print("🗄 Подключение к Qdrant...")
        engine = QdrantEngine(
            collection_name=settings.COLLECTION_NAME,
            db_path=settings.QDRANT_PATH,
            embedder=embedder,
            mode="hybrid",
            dense_vec_name="dense",
            sparse_vec_name="sparse"
        )
        return engine

    except Exception as e:
        st.error(f"Критическая ошибка при запуске системы: {e}")
        raise e


def create_user_service(engine, main_conf, rewriter_conf):
    main_llm = create_llm_instance(main_conf)
    rewriter_llm = create_llm_instance(rewriter_conf)
    return DndRagService(engine=engine, main_llm=main_llm, rewriter_llm=rewriter_llm)


try:
    bd_engine = get_engine()
    print("✅ Движок бд к работе!")
except Exception:
    st.stop()

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/8/8e/Dungeons_%26_Dragons_5th_Edition_Logo.svg", width=200)
    st.header("Настройки")

    model_names = list(MODELS_MAP.keys())

    def_main = config["defaults"]["main_llm"]
    def_rewriter = config["defaults"]["rewriter_llm"]

    selected_main_name = st.selectbox(
        "Генерация ответа",
        model_names,
        index=model_names.index(def_main) if def_main in model_names else 0
    )
    selected_rewriter_name = st.selectbox(
        "Переформулировка запроса для бд",
        model_names,
        index=model_names.index(def_rewriter) if def_rewriter in model_names else 0
    )

    if st.button("🗑 Очистить диалог", type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Статус системы:**")
    st.success("🟢 AI Engine Online")
    st.success(f"📚 База: {settings.COLLECTION_NAME}")

current_model_signature = (selected_main_name, selected_rewriter_name)

if "rag_service" not in st.session_state or st.session_state.get("last_models") != current_model_signature:
    with st.spinner("🚀 Подключение твоих нейросетей..."):
        main_conf = MODELS_MAP[selected_main_name]
        rewriter_conf = MODELS_MAP[selected_rewriter_name]

        st.session_state.rag_service = create_user_service(
            bd_engine,
            main_conf,
            rewriter_conf
        )
        st.session_state.last_models = current_model_signature

rag_service = st.session_state.rag_service

st.title("🧙‍♂️ D&D Knowledge Keeper")
st.caption("Задай вопрос Мастеру по правилам D&D 5e")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Приветствую, искатель приключений! Спрашивай о заклинаниях, монстрах или правилах."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📜 Источники ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"[{s.get('title', 'Источник')}]({s.get('url', 'https://dnd.su/')})")
                    # st.markdown("---")

if prompt := st.chat_input("Например: Как работает скрытая атака плута?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reasoning_expander = st.status("🔮 Ищу информацию в аркане...", expanded=False)
        reasoning_placeholder = reasoning_expander.empty()

        message_placeholder = st.empty()
        full_response = ""
        reasoning_content = ""
        is_thinking = False
        was_thinking = False

        history_context = [
                              {"role": m["role"], "content": m["content"]}
                              for m in st.session_state.messages
                              if m["role"] in ["user", "assistant"]
                          ][:-1]
        sources = []

        try:
            stream_generator = rag_service.generate_answer(
                query=prompt,
                chat_history=history_context,
                return_sources=True
            )

            for chunk in stream_generator:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "sources":
                        sources = chunk.get("sources", [])
                    continue

                if "<think>" in chunk:
                    is_thinking = True
                    was_thinking = True
                    chunk = chunk.replace("<think>", "")
                    reasoning_expander.update(label="🤔 Думаю...", state="running")

                if "</think>" in chunk:
                    is_thinking = False
                    was_thinking = True
                    chunk = chunk.replace("</think>", "")
                    reasoning_expander.update(label="✨ Мои думы окончены", state="complete", expanded=False)

                if is_thinking:
                    reasoning_content += chunk
                    reasoning_placeholder.markdown(reasoning_content)
                else:
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

            if not was_thinking:
                reasoning_expander.update(label="✨ Нашел ответы в аркане", state="complete", expanded=False)

            message_placeholder.markdown(full_response)

            if sources:
                with st.expander(f"📜 Источники ({len(sources)})", expanded=False):
                    for s in sources:
                        st.markdown(f"[{s.get('title', 'Источник')}]({s.get('url', 'https://dnd.su/')})")
                        # st.markdown("---")

        except ModelRateLimitError as e:
            reasoning_expander.update(label="❌ Ошибка генерации", state="error", expanded=False)
            message_placeholder.empty()

            # Показываем красивое сообщение об ошибке
            st.error(f"⛔ **Достигнут лимит запросов!** (Ошибка 429)")
            st.warning(
                f"Модель **{rag_service.main_llm.model_name}** временно недоступна. Пожалуйста, выбери другую модель в меню слева.")
            st.stop()

        except Exception as e:
            reasoning_expander.update(label="❌ Ошибка генерации", state="error", expanded=False)
            message_placeholder.empty()

            st.error(f"Произошла ошибка при генерации: {e}")
            st.warning(
                f"Извини, магические потоки нарушены. Попробуй позже.")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
