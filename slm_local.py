"""Minimal script to hit LM Studio's chat endpoint.

Example curl the script mirrors:

curl https://user-4921.nport.link/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
	"model": "qwen/qwen2.5-coder-32b",
	"system_prompt": "You answer only in rhymes.",
	"input": "What is your favorite color?"
}'
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import httpx
from dotenv import load_dotenv


def _env_default(name: str, fallback: str) -> str:
	return os.getenv(name, fallback)


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Call LM Studio's chat endpoint.")
	parser.add_argument(
		"--url",
		default=_env_default("LM_STUDIO_URL", "https://user-4921.nport.link/api/v1/chat"),
		help="Chat endpoint URL (env: LM_STUDIO_URL)",
	)
	parser.add_argument(
		"--model",
		default=_env_default("LM_STUDIO_MODEL", "qwen/qwen2.5-coder-32b"),
		help="Model identifier (env: LM_STUDIO_MODEL)",
	)
	parser.add_argument(
		"--system-prompt",
		default=_env_default("LM_STUDIO_SYSTEM_PROMPT", "You answer only in rhymes."),
		help="System prompt/persona (env: LM_STUDIO_SYSTEM_PROMPT)",
	)
	parser.add_argument(
		"--input",
		default=_env_default("LM_STUDIO_INPUT", "What is your favorite color?"),
		help="User input text (env: LM_STUDIO_INPUT)",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=float(os.getenv("LM_STUDIO_TEMPERATURE", 0.7)),
		help="Sampling temperature",
	)
	parser.add_argument(
		"--max-tokens",
		type=int,
		default=_optional_int(os.getenv("LM_STUDIO_MAX_TOKENS")),
		help="Max tokens to generate (env: LM_STUDIO_MAX_TOKENS)",
	)
	parser.add_argument(
		"--top-p",
		type=float,
		default=_optional_float(os.getenv("LM_STUDIO_TOP_P")),
		help="Nucleus sampling p (env: LM_STUDIO_TOP_P)",
	)
	parser.add_argument(
		"--api-key",
		default=os.getenv("LM_STUDIO_API_KEY"),
		help="Bearer token if your endpoint requires auth (env: LM_STUDIO_API_KEY)",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		default=float(os.getenv("LM_STUDIO_TIMEOUT", 30.0)),
		help="Request timeout in seconds",
	)
	parser.add_argument(
		"--raw",
		action="store_true",
		help="Print full JSON response instead of extracted text",
	)
	return parser


def _optional_int(value: Optional[str]) -> Optional[int]:
	return int(value) if value is not None and value.strip() else None


def _optional_float(value: Optional[str]) -> Optional[float]:
	return float(value) if value is not None and value.strip() else None


def _extract_text(payload: Any) -> Optional[str]:
	if not isinstance(payload, dict):
		return None

	if "output" in payload:
		output = payload["output"]
		if isinstance(output, str):
			return output
		if isinstance(output, list):
			parts: List[str] = []
			for item in output:
				if isinstance(item, dict):
					content = item.get("content")
					if isinstance(content, str):
						parts.append(content)
					elif isinstance(content, Iterable):
						for chunk in content:
							if isinstance(chunk, dict) and "text" in chunk:
								text_val = chunk.get("text")
								if isinstance(text_val, str):
									parts.append(text_val)
			if parts:
				return "\n".join(parts)

	if "choices" in payload and isinstance(payload["choices"], list):
		choice = payload["choices"][0]
		if isinstance(choice, dict):
			message = choice.get("message")
			if isinstance(message, dict) and isinstance(message.get("content"), str):
				return message["content"]
			if isinstance(choice.get("text"), str):
				return choice["text"]

	return None


def _build_body(args: argparse.Namespace) -> Dict[str, Any]:
	body: Dict[str, Any] = {
		"model": args.model,
		"system_prompt": args.system_prompt,
		"input": args.input,
		"temperature": args.temperature,
	}
	if args.max_tokens is not None:
		body["max_tokens"] = args.max_tokens
	if args.top_p is not None:
		body["top_p"] = args.top_p
	return body


def main() -> int:
	load_dotenv()
	parser = _build_parser()
	args = parser.parse_args()

	headers = {"Content-Type": "application/json"}
	if args.api_key:
		headers["Authorization"] = f"Bearer {args.api_key}"

	body = _build_body(args)

	try:
		with httpx.Client(timeout=args.timeout) as client:
			response = client.post(args.url, json=body, headers=headers)
			response.raise_for_status()
	except httpx.HTTPStatusError as exc:
		print(f"Request failed with status {exc.response.status_code}: {exc.response.text}")
		return 1
	except httpx.RequestError as exc:
		print(f"Request error: {exc}")
		return 1

	try:
		payload = response.json()
	except json.JSONDecodeError:
		print(response.text)
		return 0

	if args.raw:
		print(json.dumps(payload, indent=2))
		return 0

	text = _extract_text(payload)
	if text:
		print(text)
	else:
		print(json.dumps(payload, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
