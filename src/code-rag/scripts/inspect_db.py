#!/usr/bin/env python3
"""
scripts/inspect_db.py — Xem nội dung đang được lưu trong ChromaDB
"""
import argparse
import rich
from rich.console import Console
from rich.table import Table
import chromadb

def main():
    parser = argparse.ArgumentParser(description="Inspect ChromaDB contents")
    parser.add_argument("--chroma-dir", default=".chroma", help="Đường dẫn tới thư mục .chroma")
    parser.add_argument("--limit", type=int, default=5, help="Số bản ghi muốn xem (mặc định: 5)")
    parser.add_argument("--full", action="store_true", help="Hiển thị nội dung đầy đủ (không cắt gọn)")
    parser.add_argument("--output", type=str, help="Lưu toàn bộ dữ liệu ra file Markdown (ví dụ: output.md)")
    args = parser.parse_args()

    console = Console()
    
    try:
        client = chromadb.PersistentClient(path=args.chroma_dir)
        collection = client.get_collection("codebase")
    except Exception as e:
        console.print(f"[red]Không thể đọc DB (có thể DB trống hoặc chưa index). Lỗi: {e}[/red]")
        return
        
    count = collection.count()
    console.print(f"\n[bold green]📊 THỐNG KÊ CHROMADB[/bold green]")
    console.print(f"Tổng số code chunks (vectors) đang lưu: [bold cyan]{count}[/bold cyan]")
    
    if count == 0:
        return

    # Lấy ra 1 vài bản ghi mẫu để xem
    data = collection.get(limit=args.limit, include=['metadatas', 'documents'])
    
    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("# 📊 THỐNG KÊ CHROMADB\n\n")
            f.write(f"- Tổng số code chunks: **{count}**\n\n")
            f.write(f"## 🔍 Chi Tiết Bản Ghi\n\n")
            
            # Nếu có output, có thể lấy hết limit hoặc toàn bộ (ví dụ: lấy đúng limit)
            for i in range(len(data['ids'])):
                chunk_id = data['ids'][i]
                meta = data['metadatas'][i]
                doc = data['documents'][i]
                language = meta.get('language', 'python')
                
                f.write(f"### ID: `{chunk_id}`\n")
                f.write(f"- **Type:** {meta.get('chunk_type', '')}\n")
                f.write(f"- **Name:** `{meta.get('name', '')}`\n")
                f.write(f"- **File:** `{meta.get('file_path', '')}`\n")
                f.write(f"- **Lines:** {meta.get('start_line', '')} -> {meta.get('end_line', '')}\n")
                
                # Metadata
                f.write("\n<details>\n  <summary>Metadata (JSON)</summary>\n\n")
                f.write("```json\n")
                f.write(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
                f.write("```\n</details>\n\n")
                
                # Full Document
                f.write("**Document Segment:**\n")
                f.write(f"```{language}\n")
                f.write(doc + "\n")
                f.write("```\n\n")
                f.write("---\n\n")
                
        console.print(f"\n[bold green]✅ Đã lưu toàn bộ dữ liệu ra file: [white]{args.output}[/white][/bold green]")
        return
        
    console.print(f"\n[bold yellow]🔍 XEM TRƯỚC {min(count, args.limit)} BẢN GHI ĐẦU TIÊN:[/bold yellow]")
    
    if args.full:
        from rich.panel import Panel
        from rich.syntax import Syntax
        
        for i in range(len(data['ids'])):
            chunk_id = data['ids'][i]
            meta = data['metadatas'][i]
            doc = data['documents'][i]
            
            language = meta.get('language', 'python')
            
            console.print(f"\n[bold cyan]🆔 ID:[/bold cyan] {chunk_id}")
            console.print(f"[bold cyan]🏷️ Metadata:[/bold cyan] {meta}")
            console.print("[bold cyan]📄 Document Content:[/bold cyan]")
            
            syntax = Syntax(doc, language, theme="monokai", line_numbers=False, word_wrap=True)
            console.print(Panel(syntax, expand=False, border_style="blue"))
            
    else:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", width=15)
        table.add_column("Type")
        table.add_column("Name")
        table.add_column("File Path")
        table.add_column("Documents (Trích đoạn)", width=40)

        for i in range(len(data['ids'])):
            chunk_id = data['ids'][i]
            meta = data['metadatas'][i]
            doc = data['documents'][i]
            
            # Cắt ngắn doc để hiển thị
            snippet = doc.replace('\n', ' ')
            if len(snippet) > 60:
                snippet = snippet[:60] + "..."
                
            table.add_row(
                chunk_id[:10] + "...", 
                meta.get('chunk_type', ''), 
                meta.get('name', ''), 
                meta.get('file_path', ''), 
                snippet
            )
            
        console.print(table)
        console.print("\n[dim]Mẹo: Chạy `python scripts/inspect_db.py --full` để xem toàn bộ nội dung mà không bị cắt.[/dim]")
    
if __name__ == "__main__":
    main()
