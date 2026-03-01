"""Streamlit UI to run the ReAct agent for code review on a chosen folder."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, List, Set, Tuple

import streamlit as st
from dotenv import load_dotenv

from react_agent import graph
from react_agent.context import Context
from react_agent.utils import get_message_text
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

load_dotenv()

DEFAULT_MODEL = "lmstudio/qwen2.5-coder-32b"


async def _run_agent_stream(
    folder: Path,
    model: str | None,
    on_log: Callable[[str], None],
) -> Tuple[str, List[str]]:
    system_prompt = """
        Bạn là Code Debugger Agent giỏi chuyên môn.

        CÓ 4 CÔNG CỤ:
        - list_source_files: Liệt kê các file trong thư mục
        - read_file: Đọc nội dung file
        - run_python: Chạy file python
        - write_file: Ghi file

        QUY TRÌNH BẮT BUỘC:
        1. Đọc danh sách file
        2. Đọc file cần sửa
        3. Sửa lỗi
        4. Tối ưu hiệu năng code
        5. Chạy lại file để kiểm tra
        6. Lặp lại cho đến khi hết lỗi
        7. Đưa ra báo cáo cuối cùng
    """

    ctx = Context(
        model=model or DEFAULT_MODEL,
        system_prompt=system_prompt,
        base_dir=str(folder),
    )

    prompt = "Xem xét và sửa lỗi, tối ưu hiệu năng code dự án Python"

    logs: List[str] = []
    seen_ids: Set[str] = set()
    async for step in graph.astream(
        {"messages": [("user", prompt)]},
        context=ctx,
        stream_mode="values",
    ):
        msgs = step.get("messages", [])
        if not msgs:
            continue
        for msg in msgs:
            msg_id = getattr(msg, "id", None)
            if msg_id and msg_id in seen_ids:
                continue
            if msg_id:
                seen_ids.add(msg_id)

            label = "Message"
            if isinstance(msg, ToolMessage):
                label = f"Tool result ({getattr(msg, 'name', 'tool')})"
            elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                tools = ", ".join(tc.get("name", "tool") for tc in msg.tool_calls)
                label = f"Tool call -> {tools}"
            elif isinstance(msg, AIMessage):
                label = "Assistant"
            elif getattr(msg, "type", "") == "human":
                label = "User"

            content = get_message_text(msg)
            logs.append(f"{label}: {content}")
            on_log("\n".join(logs))

    final_msg = logs[-1].replace("Assistant: ", "") if logs else ""
    return final_msg, logs


def run_agent(folder: Path, model: str | None, on_log: Callable[[str], None]) -> Tuple[str, List[str]]:
    return asyncio.run(_run_agent_stream(folder, model, on_log))


def main() -> None:
    st.set_page_config(page_title="ReAct Code Review", page_icon="🧭", layout="wide")
    st.title("ReAct Agent Code Review")

    default_folder = "/Users/minhhung/Documents/GitHub/py-demo"
    folder_input = st.text_input("Folder path", default_folder, help="Path to the source folder to review")
    model_input = st.text_input("Model", DEFAULT_MODEL, help="Provider/model, e.g. lmstudio/your-model")

    if st.button("Review", type="primary"):
        folder = Path(folder_input).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            st.error(f"Folder not found: {folder}")
            return
        log_box = st.empty()
        with st.spinner("Running agent review..."):
            try:
                logs: List[str] = []

                def handle_log(chunk: str) -> None:
                    logs.append(chunk)
                    log_box.code(chunk)

                result, logs = run_agent(folder, model_input.strip() or None, handle_log)
                st.subheader("Review Result")
                st.write(result)
            except Exception as exc:  # pragma: no cover - UI path
                st.error(f"Review failed: {exc}")

if __name__ == "__main__":
    main()
