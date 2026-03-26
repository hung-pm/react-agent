import sys
from pathlib import Path

import asyncio
import os
import json
from unittest.mock import patch, MagicMock

# Import the tool from the local package
from react_agent.tools import get_git_changes

async def main():
    # Since get_git_changes uses langgraph's get_runtime(Context),
    # which requires to be called inside a LangGraph node,
    # we mock it here so we can test the tool independently in the console.
    with patch("react_agent.tools.get_runtime") as mock_get_runtime:
        # Create a fake Context pointing to the current directory
        mock_runtime_instance = MagicMock()
        mock_runtime_instance.context.base_dir = os.getcwd()
        mock_get_runtime.return_value = mock_runtime_instance
        
        print("Testing get_git_changes() tool...\n")
        
        try:
            result = await get_git_changes()
            print("=== Kết quả (JSON Format) ===")
            print(json.dumps(result, indent=4, ensure_ascii=False))
        except Exception as e:
            print(f"Lỗi hệ thống: {e}")

if __name__ == "__main__":
    asyncio.run(main())
