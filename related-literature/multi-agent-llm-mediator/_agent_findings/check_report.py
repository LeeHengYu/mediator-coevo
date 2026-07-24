#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrity checks for the generated report.md."""
import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
txt = open(os.path.join(BASE, "report.md"), encoding="utf-8").read()
toc = txt.split("\n---\n", 1)[0]

print("TOC entries:", len(re.findall(r"^\d+\. \[.*?\]\(#item-", toc, flags=re.M)))
print("Body headers:", len(re.findall(r"^### \d+\. ", txt, flags=re.M)))
anchors = set(re.findall(r'<a id="(item-[^"]+)"></a>', txt))
targets = set(re.findall(r"\]\(#(item-[^)]+)\)", txt))
print("Anchors:", len(anchors), "| TOC targets:", len(targets),
      "| unresolved:", len(targets - anchors))
legend = "fields.yaml"
print("Leaked [不确定] (excl legend):",
      sum(1 for line in txt.splitlines() if "[不确定]" in line and legend not in line))
print("cluster ? in body headers:", txt.count("cluster **?**"))
print(".json shown as TOC name:", len(re.findall(r"\[[^\]]*\.json\]\(#item-", toc)))
print("blank-tier TOC tags:", len(re.findall(r"`[A-H]·\s+·", toc)))
hdrs = re.findall(r"^### ([A-H]) —", toc, flags=re.M)
print("cluster section headers:", hdrs)
print("report size MB:", round(os.path.getsize(os.path.join(BASE, "report.md")) / 1024 / 1024, 2))
