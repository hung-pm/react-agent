#!/usr/bin/env python3
"""
scripts/index.py — Scan codebase và index vào ChromaDB

Chạy một lần trước khi dùng agent.

Usage:
    python scripts/index.py --dir ./your-project
    python scripts/index.py --dir ./your-project --reindex
"""
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import print as rprint

load_dotenv()

# Thêm root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunker import chunk_directory
from src.vector_store import CodebaseVectorStore
from src.code_graph import CodeGraph

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Index codebase vào ChromaDB"
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Đường dẫn thư mục codebase cần index",
    )
    parser.add_argument(
        "--chroma-dir",
        default=".chroma",
        help="Thư mục lưu ChromaDB (default: .chroma)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Xóa index cũ và index lại toàn bộ",
    )
    args = parser.parse_args()

    target_dir = Path(args.dir).resolve()
    if not target_dir.exists():
        console.print(f"[red]ERROR:[/red] Thư mục không tồn tại: {target_dir}")
        sys.exit(1)

    console.rule("[bold blue]Codebase Indexer")
    console.print(f"  Target dir : [cyan]{target_dir}[/cyan]")
    console.print(f"  ChromaDB   : [cyan]{args.chroma_dir}[/cyan]")
    console.print(f"  Reindex    : [cyan]{args.reindex}[/cyan]\n")

    # ── Step 1: Chunking ──────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing codebase...", total=None)
        t0 = time.time()
        chunks = chunk_directory(str(target_dir))
        elapsed = time.time() - t0
        progress.update(task, description=f"Parsed {len(chunks)} chunks in {elapsed:.1f}s")

    if not chunks:
        console.print("[yellow]Không tìm thấy code nào. Kiểm tra lại --dir và extension hỗ trợ.[/yellow]")
        sys.exit(0)

    # Thống kê chunks
    from collections import Counter
    by_lang = Counter(c.language for c in chunks)
    by_type = Counter(c.chunk_type for c in chunks)

    table = Table(title="Chunks found", show_header=True)
    table.add_column("Language", style="cyan")
    table.add_column("Count", justify="right")
    for lang, count in by_lang.most_common():
        table.add_row(lang, str(count))
    console.print(table)

    table2 = Table(title="Chunk types", show_header=True)
    table2.add_column("Type", style="green")
    table2.add_column("Count", justify="right")
    for t, count in by_type.most_common():
        table2.add_row(t, str(count))
    console.print(table2)

    # ── Step 2: Build Code Graph ──────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building call graph...", total=None)
        graph = CodeGraph()
        graph.build(chunks)
        stats = graph.summary()
        progress.update(
            task,
            description=f"Graph: {stats['nodes']} nodes, {stats['edges']} edges",
        )

    console.print(
        f"  [green]✓[/green] Call graph: "
        f"[bold]{stats['nodes']}[/bold] functions, "
        f"[bold]{stats['edges']}[/bold] call edges"
    )

    # ── Step 3: Embedding + Index vào ChromaDB ────────────────────────
    store = CodebaseVectorStore(persist_dir=args.chroma_dir)

    if args.reindex:
        console.print("  [yellow]Clearing old index...[/yellow]")
        store.clear()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding & indexing...", total=len(chunks))

        # Index theo batch, cập nhật progress
        batch_size = 50
        total_indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            # Gọi trực tiếp để update progress sau mỗi batch
            indexed = store.index_chunks(batch, batch_size=batch_size)
            total_indexed += indexed
            progress.update(task, advance=len(batch))

    final_stats = store.get_stats()
    console.print(
        f"\n  [green]✓[/green] Done! "
        f"[bold]{final_stats['total_chunks']}[/bold] chunks in ChromaDB "
        f"→ [cyan]{args.chroma_dir}[/cyan]"
    )
    console.print("\n[bold green]Index complete! Bạn có thể chạy agent ngay bây giờ.[/bold green]")


if __name__ == "__main__":
    main()
