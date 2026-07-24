#!/usr/bin/env python3
"""Extract the synthesized findings (longest assistant text block) from a
background-agent JSONL transcript. Usage: python3 extract.py <transcript.output>
"""
import json
import sys


def main(path):
    asst_texts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            msg = obj.get('message') if isinstance(obj, dict) else None
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                content = msg.get('content')
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get('type') == 'text' and b.get('text', '').strip():
                            asst_texts.append(b['text'])
    # The synthesized report is the longest assistant text block.
    asst_texts.sort(key=len, reverse=True)
    print(asst_texts[0] if asst_texts else 'NO ASSISTANT TEXT FOUND')

if __name__ == '__main__':
    main(sys.argv[1])
