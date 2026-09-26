import streamlit as st
from rag import retrieve_and_rerank, generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Mining Unit 1 RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.markdown("### 🤖 Language Model")
    st.write("Llama 3.2 3B")

    st.markdown("### 🔎 Embedding Model")
    st.write("all-MiniLM-L6-v2")

    st.markdown("### 🗄️ Vector Database")
    st.write("ChromaDB")

    st.markdown("### 📦 Knowledge Base")
    st.write("85 document chunks")

    st.markdown("### 🔄 Retrieval Pipeline")
    st.write("Top 10 → Cross-Encoder → Top 5")

    st.divider()

    st.markdown("### 📚 Project")
    st.write("Data Mining Unit 1")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN TITLE
# ============================================================

st.title("📚 Data Mining Unit 1 RAG Assistant")

st.caption(
    "Ask questions based on the provided Data Mining Unit 1 materials."
)


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show sources only for assistant messages
        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander("📖 Retrieved Sources"):

                for source in message["sources"]:

                    st.markdown(
                        f"**{source['source']} — "
                        f"Slide {source['slide']}**"
                    )

                    st.caption(
                        f"Reranker score: {source['score']:.4f}"
                    )

                    st.write(source["text"])


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask a question about Data Mining Unit 1..."
)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if query:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # Retrieve and rerank
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Retrieving relevant information..."
        ):

            ranked_results = retrieve_and_rerank(
                query,
                initial_k=10,
                final_k=5
            )


        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = generate_answer(
                query,
                ranked_results
            )


        # ----------------------------------------------------
        # Prepare sources
        # ----------------------------------------------------

        sources = []

        for score, document, metadata in ranked_results:

            sources.append(
                {
                    "source": metadata["source"],
                    "slide": metadata["slide"],
                    "score": float(score),
                    "text": document
                }
            )


        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.markdown(answer)


        # ----------------------------------------------------
        # Display retrieved sources
        # ----------------------------------------------------

        with st.expander("📖 Retrieved Sources"):

            for source in sources:

                st.markdown(
                    f"**{source['source']} — "
                    f"Slide {source['slide']}**"
                )

                st.caption(
                    f"Reranker score: "
                    f"{source['score']:.4f}"
                )

                st.write(source["text"])


    # --------------------------------------------------------
    # Save assistant response to chat history
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )