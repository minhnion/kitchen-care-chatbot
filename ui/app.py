# ui/app.py

import streamlit as st
import requests
import uuid

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Kitchen Care Chatbot",
    page_icon="🤖"
)

st.title("🤖 Kitchen Care - Trợ lý Mua sắm")
st.caption("Hỏi tôi bất cứ điều gì về các sản phẩm nhà bếp của chúng tôi!")

# --- URL của API Backend (FastAPI) ---
# Quan trọng: Đảm bảo backend FastAPI của bạn đang chạy!
# Nếu bạn chạy cả hai trên cùng một máy, đây là URL mặc định.
API_URL = "http://127.0.0.1:8000/api/v1/chatbot/chat"

# --- Quản lý Trạng thái Phiên (Session State) ---
# Dùng session_state của Streamlit để lưu lịch sử chat và session_id

# Khởi tạo session_id nếu chưa có
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Khởi tạo lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?"}
    ]

# --- Hiển thị Lịch sử Chat ---
# Lặp qua danh sách tin nhắn đã lưu và hiển thị chúng
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # Nếu là tin nhắn của assistant và có source_documents, hiển thị chúng
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("Xem nguồn tham khảo"):
                for source in msg["sources"]:
                    st.json(source) # Hiển thị JSON của source_documents

# --- Xử lý Input của Người dùng ---
# Dùng st.chat_input để tạo một ô nhập liệu ở cuối trang
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # 1. Hiển thị tin nhắn của người dùng lên giao diện ngay lập tức
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Chuẩn bị dữ liệu để gửi đến API
    # Lấy lịch sử chat (trừ tin nhắn chào mừng đầu tiên)
    chat_history_for_api = [
        {"type": "human" if msg["role"] == "user" else "ai", "content": msg["content"]}
        for msg in st.session_state.messages[1:-1] # Bỏ tin nhắn chào mừng và tin nhắn hiện tại
    ]
    
    payload = {
        "message": prompt,
        "session_id": st.session_state.session_id,
        "chat_history": chat_history_for_api
    }

    # 3. Gọi API và xử lý response
    try:
        with st.spinner("Bot đang suy nghĩ..."):
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Ném lỗi nếu status code là 4xx hoặc 5xx
            
            data = response.json()
            answer = data.get("answer", "Xin lỗi, đã có lỗi xảy ra.")
            sources = data.get("source_documents", [])
            
            # 4. Hiển thị câu trả lời của bot
            msg_to_add = {"role": "assistant", "content": answer, "sources": sources}
            st.session_state.messages.append(msg_to_add)
            
            with st.chat_message("assistant"):
                st.write(answer)
                if sources:
                    with st.expander("Xem nguồn tham khảo"):
                        for source in sources:
                            st.json(source) # Hiển thị JSON của source_documents

    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối đến API: {e}")
        error_msg = {"role": "assistant", "content": f"Xin lỗi, tôi không thể kết nối đến máy chủ. Vui lòng thử lại sau. (Lỗi: {e})"}
        st.session_state.messages.append(error_msg)
        with st.chat_message("assistant"):
            st.write(error_msg["content"])