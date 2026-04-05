#!/usr/bin/env python3
"""
scripts/inspect_graph.py — Xem các mối quan hệ Caller / Callee trong CodeGraph
"""
import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Thêm thư mục gốc vào path để import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.chunker import chunk_directory
from src.parsing.code_graph import CodeGraph

def main():
    parser = argparse.ArgumentParser(description="Inspect CodeGraph Relationships")
    parser.add_argument("--dir", required=True, help="Đường dẫn codebase (ví dụ: ./sample_project)")
    parser.add_argument("--node", type=str, help="Xem chi tiết quan hệ của 1 hàm/class cụ thể")
    
    args = parser.parse_args()
    console = Console()
    
    target_dir = Path(args.dir).resolve()
    if not target_dir.exists():
        console.print(f"[red]Không tìm thấy thư mục: {target_dir}[/red]")
        sys.exit(1)
        
    console.print(f"[dim]Đang quét và dựng lại đồ thị cho {target_dir}...[/dim]")
    chunks = chunk_directory(str(target_dir))
    graph = CodeGraph()
    graph.build(chunks)
    
    stats = graph.summary()
    console.print(f"\n[bold green]📊 THỐNG KÊ CODEGRAPH[/bold green]")
    console.print(f"- Số Nodes (Hàm/Class): [bold blue]{stats['nodes']}[/bold blue]")
    console.print(f"- Số Edges (Đường nối gọi hàm): [bold blue]{stats['edges']}[/bold blue]\n")
    
    if args.node:
        if args.node not in graph.G:
            console.print(f"[red]Không tìm thấy tên '{args.node}' trong đồ thị![/red]")
            return
            
        callers = graph.get_callers(args.node)
        callees = graph.get_callees(args.node)
        
        console.print(f"[bold yellow]🔍 CHI TIẾT NODE: [white]{args.node}[/white][/bold yellow]")
        
        console.print("\n[cyan]← Ai đang gọi tới nó (Callers)?[/cyan]")
        if callers:
            for c in callers:
                console.print(f"    • {c}")
        else:
            console.print("    [dim](Không có hàm nào trong nội bộ dự án gọi)[/dim]")
            
        console.print("\n[magenta]→ Nó đang gọi tới ai (Callees)?[/magenta]")
        if callees:
            for c in callees:
                console.print(f"    • {c}")
        else:
            console.print("    [dim](Không gọi hàm nội bộ nào khác)[/dim]")
            
        impacts = graph.get_impact(args.node)
        console.print(f"\n[yellow]⚠️ Phạm vi ảnh hưởng (Impact Analysis)[/yellow]")
        console.print("[dim]  (Nếu hàm này đổi tên/báo lỗi, các hàm sau ở tầng trên cũng sẽ gãy)[/dim]")
        if impacts:
            for i in impacts:
                console.print(f"    • {i}")
        else:
             console.print("    [dim](Không có)[/dim]")
             
    else:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Hàm / Class (Node)")
        table.add_column("Các hàm mà nó dùng (Callees) →", style="magenta")
        table.add_column("Các hàm đang chạy nó (Callers) ←", style="cyan")
        
        for node in graph.G.nodes:
            callers = graph.get_callers(node)
            callees = graph.get_callees(node)
            
            # Ẩn những hàm mồ côi (không gọi ai và cũng không ai gọi)
            if not callers and not callees:
                continue
                
            caller_str = "\n".join([f"• {c}" for c in callers]) if callers else "[dim]none[/dim]"
            callee_str = "\n".join([f"• {c}" for c in callees]) if callees else "[dim]none[/dim]"
            
            table.add_row(f"[bold]{node}[/bold]", callee_str, caller_str)
            table.add_section()
            
        console.print(table)
        console.print("\n[dim]Mẹo: Chạy `python scripts/inspect_graph.py --dir ./sample_project --node tên_hàm` để xem chi tiết 1 nhánh.[/dim]")

if __name__ == "__main__":
    main()
