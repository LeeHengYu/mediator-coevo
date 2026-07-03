# Task Instruction

Execute the following Python script to produce `/root/renewal_playbook_updated.hwpx`.

```python
import zipfile, json, csv, os, copy, io, re
from lxml import etree

# ── 1. Load update sources ──────────────────────────────────────────────
with open('/root/renewal_update.json', 'r', encoding='utf-8') as f:
    updates = json.load(f)

followups = []
with open('/root/followups.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        followups.append(row)
# Sort by 'sequence' field (integer sort)
followups.sort(key=lambda r: int(r['sequence']))

print('Update JSON:', json.dumps(updates, ensure_ascii=False, indent=2))
print('Follow-ups (sorted):', followups)

# ── 2. Open the original HWPX (ZIP archive) ────────────────────────────
src_path = '/root/renewal_playbook.hwpx'
dst_path = '/root/renewal_playbook_updated.hwpx'

with zipfile.ZipFile(src_path, 'r') as zin:
    member_names = zin.namelist()
    member_data = {}
    for name in member_names:
        member_data[name] = zin.read(name)

print('Archive members:', member_names)

# ── 3. Identify section XML files to edit ───────────────────────────────
# Typically Contents/section*.xml  (could be section0.xml, section1.xml, etc.)
xml_members = [n for n in member_names if n.lower().endswith('.xml')]
print('XML members:', xml_members)

# We'll process every XML file that contains <hp:p> elements
NS = {
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hp_char': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
}

def get_para_text(p_elem):
    """Concatenate all <hp:t> text within an <hp:p>."""
    texts = []
    for t in p_elem.iter():
        if t.tag.endswith('}t') or t.tag == 't':
            if t.text:
                texts.append(t.text)
    return ''.join(texts)

def set_para_text(p_elem, new_text):
    """Rewrite paragraph: put all text into the first <hp:t>, clear others."""
    t_nodes = [t for t in p_elem.iter() if t.tag.endswith('}t') or t.tag == 't']
    if not t_nodes:
        return
    t_nodes[0].text = new_text
    for t in t_nodes[1:]:
        t.text = ''

def remove_linesegarray(p_elem):
    """Remove <hp:lineSegArray> or <hp:linesegarray> (any case) children."""
    to_remove = []
    for child in p_elem:
        local = etree.QName(child.tag).localname.lower()
        if local == 'linesegarray':
            to_remove.append(child)
    for child in to_remove:
        p_elem.remove(child)

# ── 4. Build replacement map from JSON ──────────────────────────────────
# The JSON typically has old→new pairs or field values.
# We need to discover what the old values are by inspecting the document.
# First, let's look at the JSON structure.
print('\n--- Inspecting update JSON keys ---')
for k, v in updates.items():
    print(f'  {k}: {v}')

# ── 5. Parse all section XMLs, discover old values, build replacements ──
# Strategy: parse the document first to understand its content, then apply.

section_files = [n for n in member_names if 'section' in n.lower() and n.lower().endswith('.xml')]
if not section_files:
    # fallback: any xml under Contents/
    section_files = [n for n in member_names if 'contents/' in n.lower() and n.lower().endswith('.xml')]
print('Section files to process:', section_files)

# First pass: print all paragraph texts for debugging
for sf in section_files:
    tree = etree.fromstring(member_data[sf])
    print(f'\n=== {sf} paragraphs ===')
    for i, p in enumerate(tree.iter()):
        if p.tag.endswith('}p') or p.tag == 'p':
            txt = get_para_text(p)
            if txt.strip():
                print(f'  [{i}] {txt}')

# ── 6. Build old→new replacement pairs ──────────────────────────────────
# The JSON may have structure like {"customer_name": {"old": ..., "new": ...}, ...}
# or flat {"customer_name": "new_value", ...}
# Let's handle both.

replacements = {}  # old_string -> new_string

# Detect structure
sample_val = next(iter(updates.values()))
if isinstance(sample_val, dict) and 'old' in sample_val and 'new' in sample_val:
    # Structured with old/new
    for key, val in updates.items():
        old = val.get('old', '')
        new = val.get('new', '')
        if old and old != new:
            replacements[old] = new
else:
    # Flat: we need to find old values from the document text
    # Collect all paragraph texts
    all_texts = []
    for sf in section_files:
        tree = etree.fromstring(member_data[sf])
        for p in tree.iter():
            if p.tag.endswith('}p') or p.tag == 'p':
                txt = get_para_text(p)
                if txt.strip():
                    all_texts.append(txt)
    
    # Try to match JSON keys to document patterns
    # This is a heuristic; the structured format is more reliable
    print('\nWARNING: Flat JSON detected. Will attempt pattern matching.')
    for key, new_val in updates.items():
        replacements[key] = str(new_val)

print('\nReplacements:', json.dumps(replacements, ensure_ascii=False, indent=2))

# ── 7. Identify follow-up lines ─────────────────────────────────────────
# We need to find the 3 existing follow-up lines and replace them.
# The follow-up action text from CSV
followup_texts = []
for row in followups:
    # Try common column names
    text = row.get('action') or row.get('item') or row.get('text') or row.get('description') or row.get('followup') or ''
    followup_texts.append(text.strip())

print('Follow-up replacement texts:', followup_texts)

# ── 8. Process each section XML ─────────────────────────────────────────
APPENDIX_SENTINEL = '이 부록 문단은 그대로 유지해야 합니다.'

for sf in section_files:
    tree = etree.fromstring(member_data[sf])
    
    # Collect all <hp:p> elements
    all_paras = [p for p in tree.iter() if p.tag.endswith('}p') or p.tag == 'p']
    
    # --- Phase A: Find follow-up lines ---
    # Heuristic: follow-up lines are consecutive paragraphs that will be replaced.
    # We look for a block of exactly 3 lines that look like follow-up items.
    # They might contain numbering like 1. 2. 3. or similar patterns.
    followup_indices = []
    for i, p in enumerate(all_paras):
        txt = get_para_text(p).strip()
        # Check if this looks like a follow-up line (numbered or bulleted)
        # Common patterns: starts with digit+period, or contains 후속/follow
        if re.match(r'^\d+[.\)]\s', txt) or re.match(r'^[①②③④⑤]', txt) or re.match(r'^[-•]\s', txt):
            followup_indices.append(i)
    
    print(f'\nPotential follow-up indices in {sf}:', followup_indices)
    for idx in followup_indices:
        print(f'  [{idx}] {get_para_text(all_paras[idx]).strip()}')
    
    # If we found exactly 3 consecutive follow-up lines, replace them
    # If not exactly 3, try to find a consecutive block of 3
    if len(followup_indices) >= 3:
        # Find the first consecutive block of 3
        block_start = None
        for j in range(len(followup_indices) - 2):
            if (followup_indices[j+1] == followup_indices[j] + 1 and
                followup_indices[j+2] == followup_indices[j] + 2):
                block_start = j
                break
        
        if block_start is not None:
            fu_para_indices = followup_indices[block_start:block_start+3]
        else:
            # Take first 3
            fu_para_indices = followup_indices[:3]
        
        print(f'Replacing follow-up paragraphs at indices: {fu_para_indices}')
        for k, idx in enumerate(fu_para_indices):
            if k < len(followup_texts):
                old_txt = get_para_text(all_paras[idx]).strip()
                # Preserve the numbering prefix if present
                old_match = re.match(r'^(\d+[.\)]\s*|[①②③④⑤]\s*|[-•]\s*)', old_txt)
                new_match = re.match(r'^(\d+[.\)]\s*|[①②③④⑤]\s*|[-•]\s*)', followup_texts[k])
                
                if new_match:
                    # CSV text already has numbering
                    new_text = followup_texts[k]
                elif old_match:
                    # Preserve old numbering prefix
                    new_text = old_match.group(1) + followup_texts[k]
                else:
                    new_text = followup_texts[k]
                
                print(f'  Follow-up [{idx}]: "{old_txt}" -> "{new_text}"')
                set_para_text(all_paras[idx], new_text)
                remove_linesegarray(all_paras[idx])
    
    # --- Phase B: Apply text replacements ---
    for p in all_paras:
        txt = get_para_text(p)
        if not txt.strip():
            continue
        
        # Skip appendix sentinel
        if APPENDIX_SENTINEL in txt:
            print(f'  SKIPPING appendix paragraph: {txt.strip()[:60]}...')
            continue
        
        modified = False
        new_txt = txt
        for old_val, new_val in replacements.items():
            if old_val in new_txt:
                new_txt = new_txt.replace(old_val, new_val)
                modified = True
        
        if modified:
            print(f'  Replaced in para: "{txt.strip()[:80]}" -> "{new_txt.strip()[:80]}"')
            set_para_text(p, new_txt)
            remove_linesegarray(p)
    
    # Serialize back
    member_data[sf] = etree.tostring(tree, xml_declaration=True, encoding='UTF-8')

# ── 9. Write the updated HWPX ───────────────────────────────────────────
with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name in member_names:
        zout.writestr(name, member_data[name])

print(f'\nWrote {dst_path}')
print('Done.')
```

After running the script, verify:
1. The output file exists at `/root/renewal_playbook_updated.hwpx`.
2. It is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/renewal_playbook_updated.hwpx','r'); print(z.namelist()); z.close()"`
3. Inspect the section XML(s) to confirm:
   - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
   - Old values from the JSON are no longer present.
   - New values from the JSON appear in the correct locations.
   - Follow-up lines match the CSV items in sequence order.
   - No `<hp:lineSegArray>` (or `<hp:linesegarray>`) elements remain in any paragraph whose text was modified.

If the initial script's heuristics for finding follow-up lines or replacement values don't match the actual document structure (e.g., the JSON is flat instead of old/new pairs, or follow-ups aren't numbered), inspect the printed debug output carefully and adjust the approach accordingly. The key principles:
- Concatenate all `<hp:t>` text within each `<hp:p>` before matching.
- Rewrite text into the first `<hp:t>` node, clear subsequent ones.
- Always remove `<hp:lineSegArray>` from any modified paragraph.
- Preserve the appendix sentinel paragraph untouched.
- Maintain valid ZIP structure with all original members.

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.