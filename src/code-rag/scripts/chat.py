#!/usr/bin/env python3
"""
scripts/chat.py — Giao diện CLI để chat với codebase agent

Usage:
    python scripts/chat.py --dir ./your-project
    python scripts/chat.py --dir ./your-project --model gpt-4o
"""
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import chunk_directory
from src.vector_store import CodebaseVectorStore
from src.code_graph import CodeGraph
from src.agent import create_agent

console = Console()


def load_codebase(target_dir: str, chroma_dir: str):
    """Load chunks, build graph, connect vector store."""

    # ── Chunks ────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Loading codebase...", total=None)
        chunks = chunk_directory(target_dir)
        p.update(task, description=f"Loaded {len(chunks)} chunks")

    if not chunks:
        console.print("[red]No code chunks found. Run index.py first or check --dir.[/red]")
        sys.exit(1)

    # ── Code Graph ────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task("Building call graph...", total=None)
        graph = CodeGraph()
        graph.build(chunks)
        stats = graph.summary()
        p.update(task, description=f"Graph ready: {stats['nodes']} nodes")

    # ── Vector Store ──────────────────────────────────────────────────
    store = CodebaseVectorStore(persist_dir=chroma_dir)
    db_stats = store.get_stats()

    if db_stats["total_chunks"] == 0:
        console.print("[yellow]Vector store trống. Đang index...[/yellow]")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
            task = p.add_task("Indexing...", total=None)
            count = store.index_chunks(chunks)
            p.update(task, description=f"Indexed {count} chunks")
    else:
        console.print(
            f"  [green]✓[/green] Vector store: "
            f"[bold]{db_stats['total_chunks']}[/bold] chunks cached"
        )

    return store, graph


def stream_response(app, messages: list, console: Console) -> str:
    """Gọi agent và stream response ra console."""
    full_response = ""
    tool_calls_shown = set()

    with console.status("[bold blue]Agent đang suy nghĩ...[/bold blue]"):
        result = app.invoke({"messages": messages})

    # Lấy tất cả messages mới
    new_messages = result["messages"]

    # In ra tool calls và final answer
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            # In tool calls nếu có
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    call_key = f"{tc['name']}:{tc['args']}"
                    if call_key not in tool_calls_shown:
                        tool_calls_shown.add(call_key)
                        args_str = str(tc["args"])[:80]
                        console.print(
                            f"  [dim]→ tool:[/dim] [cyan]{tc['name']}[/cyan]"
                            f"[dim]({args_str})[/dim]"
                        )
            # Final text response
            if msg.content and isinstance(msg.content, str) and msg.content.strip():
                full_response = msg.content

    return full_response, result["messages"]


def main():
    parser = argparse.ArgumentParser(description="Chat với codebase agent")
    parser.add_argument("--dir", required=True, help="Đường dẫn codebase")
    parser.add_argument("--chroma-dir", default=".chroma", help="ChromaDB dir")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    if not target_dir.exists():
        console.print(f"[red]Thư mục không tồn tại: {target_dir}[/red]")
        sys.exit(1)

    console.rule("[bold blue]Codebase Bug-Fix Agent")
    console.print(f"  Codebase  : [cyan]{target_dir}[/cyan]")
    console.print(f"  Model     : [cyan]{args.model}[/cyan]")
    console.print()

    # Load
    store, graph = load_codebase(str(target_dir), args.chroma_dir)

    # Create agent with hybrid search
    app = create_agent(store, graph, model=args.model)

    console.print("\n[bold green]Agent sẵn sàng![/bold green]")
    console.print("Gõ câu hỏi về codebase hoặc mô tả bug. Gõ [bold]exit[/bold] để thoát.\n")

    # Conversation history
    conversation_messages = []

    while True:
        try:
            user_input = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # Thêm vào history
        conversation_messages.append(HumanMessage(content=user_input))

        # Gọi agent
        response_text, updated_messages = stream_response(
            app, conversation_messages, console
        )

        # Cập nhật history với tất cả messages mới
        conversation_messages = updated_messages

        # In response
        if response_text:
            console.print()
            console.print(Panel(
                Markdown(response_text),
                title="[bold blue]Agent[/bold blue]",
                border_style="blue",
            ))
        console.print()


if __name__ == "__main__":
    main()
