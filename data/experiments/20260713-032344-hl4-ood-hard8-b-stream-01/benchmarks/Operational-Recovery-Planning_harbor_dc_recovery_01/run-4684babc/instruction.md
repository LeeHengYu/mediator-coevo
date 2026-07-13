# Task Instruction

## Task: Build Server Provisioning Recovery Plan

You must create two files:
1. `/root/server_provisioning_recovery_plan_analysis.xlsx`
2. `/root/server_provisioning_recovery_summary.md`

Use `/root/Open_Server_Requests_Listing.xlsx` as reference context (read it first to understand the data landscape).

### Step 0: Read the reference file
```bash
cd /root
python3 -c "
import openpyxl
wb = openpyxl.load_workbook('Open_Server_Requests_Listing.xlsx', data_only=True)
for s in wb.sheetnames:
    ws = wb[s]
    print(f'Sheet: {s}, rows={ws.max_row}, cols={ws.max_column}')
    for r in range(1, min(6, ws.max_row+1)):
        print([ws.cell(r,c).value for c in range(1, min(15, ws.max_column+1))])
"
```

### Step 1: Create the Excel workbook with openpyxl

Write a Python script that creates `/root/server_provisioning_recovery_plan_analysis.xlsx` with exactly 3 sheets named:
- `Current Capacity and Racks`
- `Relocated Network Equipment`
- `10 hr Shift Relocate Network Eq`

#### Date setup
- 100 calendar days from 2018-01-22 (row 4) to 2018-05-01 (row 103), one day per row.
- Column A: row labels (optional but fine to leave blank)
- Column B: dates. Row 4 = date(2018,1,22). Rows 5-103: use formula `=B{prev}+1` so they are formulas.

#### Header layout (same on ALL 3 sheets)
- C2: `Rack-Mount Web Servers`
- F2: `Blade Database Servers`
- I2: `Network Appliances`
- Row 3, columns C through K:
  - C3: `Planned Production`
  - D3: `Purchase Orders Due`
  - E3: `Cumulative Open Purchase Orders (EOD)`
  - F3: `Planned Production`
  - G3: `Purchase Orders Due`
  - H3: `Cumulative Open Purchase Orders (EOD)`
  - I3: `Actual Var to PO`
  - J3: `Total Prod`
  - K3: `Notes`

#### PO Due quantities (columns D and G) — same on all 3 sheets
These are numeric constants:
- 2018-01-22: D=1065, G=855
- 2018-02-01: D=855, G=555
- 2018-02-15: D=900, G=900
- 2018-03-01: D=900, G=575
- 2018-04-02: D=900, G=575
- 2018-05-01: D=900, G=575
- All other dates: D=0, G=0

#### Column types
- Columns C, D, F, G, I: numeric constants (integers), no formulas.
- Column E (Cumulative Open PO for Web): formula. For row 4: `=D4-C4`. For rows 5+: `=E{prev}+D{row}-C{row}`. This tracks cumulative PO minus cumulative production.
- Column H (Cumulative Open PO for DB): formula. For row 4: `=G4-F4`. For rows 5+: `=H{prev}+G{row}-F{row}`.
- Column J (Total Prod): formula. `=C{row}+F{row}+I{row}`.

#### Weekend and holiday rules
Weekends (Saturday=5, Sunday=6 in Python's weekday()) and Manitoba holidays (2018-02-19 Family Day, 2018-03-30 Good Friday): set C=0, F=0, I=0.

#### Sheet 1: Current Capacity and Racks
- Web planned production (col C): 0 on non-working days. On working days: <=120 before 2018-02-05, <=135 on/after 2018-02-05.
- DB planned production (col F): 0 before 2018-03-01 (DB start constraint). On working days on/after 2018-03-01: <=120 before 2018-02-05 (N/A since 03-01 > 02-05), so <=135.
- Network Appliances (col I): must sum to at least 1200 across rows 4-103. Spread across working days. Use ~18-20 per working day to reach 1200+.
- Strategy for Web: distribute production to try to keep up with POs. Total POs for web = 1065+855+900+900+900+900 = 5520. With ~68 working days and max 135 (some at 120), max web ≈ 120*10 + 135*58 = 1200+7830 = 9030, so feasible. Spread evenly: use 120 on working days before Feb 5, 135 on working days Feb 5+.
- Strategy for DB: starts Mar 1. Total DB POs = 855+555+900+575+575+575 = 4035. Working days Mar 1 to May 1 ≈ 43. At 135/day max = 5805, feasible. Use 135 on working days.
- For this scenario, by May 1 the cumulative Web production likely won't cover all 5520 POs (let's check: ~10 working days before Feb 5 at 120 = 1200, ~58 working days Feb 5 to May 1 at 135 = 7830, total = 9030 > 5520, so Web CAN finish). DB: ~43 working days at 135 = 5805 > 4035, so DB can also finish. But the requirement says May PO On-Time: No. So we need to set production levels that make it NOT on-time. Let me reconsider: use moderate production levels that are realistic but insufficient. Use Web ~80/day and DB ~80/day (within limits). Actually, re-reading: the constraint says planned production must be <= 120/135, not that it must equal that. To get "No" for on-time, use lower rates. Let's target: Web ~75/day avg, DB ~60/day avg (starting Mar 1).
  - Web: 75/day * 68 working days ≈ 5100 < 5520. So May PO not on time.
  - DB: 60/day * 43 working days ≈ 2580 < 4035. Not on time either.
  - Use Web=75 on working days (before Feb 5: 75, after: 75). DB=60 on working days from Mar 1.
  - Network: 18/working day * 68 ≈ 1224 >= 1200. Use 18.

#### Sheet 2: Relocated Network Equipment
- Same capacity limits: Web/DB <=120 before Feb 5, <=135 on/after Feb 5.
- DB production: cannot start before 2018-02-20. So DB=0 before Feb 20.
- Network Appliances: at least 100 total before 2018-02-01, and 0 on/after 2018-02-01.
  - Working days Jan 22-31: Jan 22(Mon),23,24,25,26(Fri) = 5 days; Jan 29,30,31 = 3 days. Total 8 working days before Feb 1.
  - Network = 13/day for 8 days = 104 >= 100. Use 13.
  - Network = 0 from Feb 1 onward.
- May PO On-Time: Web Yes, Database No.
  - Web must finish 5520 by May 1. Working days ~68. Need avg ~82/day. Use 100 before Feb 5 (~10 days = 1000), 120 after (~58 days = 6960), total = 7960 > 5520. But we need it to be exactly on-time (Yes). Use 120 before Feb 5, 80 after Feb 5. 10*120=1200, 58*80=4640, total=5840>5520. Yes.
  - DB must NOT finish 4035. Starts Feb 20. Working days Feb 20 to May 1 ≈ 50. At 60/day = 3000 < 4035. Use DB=60.

#### Sheet 3: 10 hr Shift Relocate Network Eq
- Network Appliances = 0 for entire horizon.
- DB cannot start before 2018-02-20.
- There must be a "temporary 10-hour shift window" of 20-24 working days on/after Feb 1 where Web > 135 OR Database > 135 (each such day counts). On those days, individual values <= 170. Outside the window: <=120 before Feb 5, <=135 on/after Feb 5.
- May PO On-Time: Yes (both Web and DB).
  - Web needs 5520 total. DB needs 4035 total.
  - Web: ~10 working days before Feb 5 at 120 = 1200. Then we need 4320 more in ~58 working days. At 135 = 7830 normally, but let's use the 10hr window. Use 22 days at 170 = 3740, remaining 36 days at 135 = 4860. Total after Feb 5 = 8600. Grand total = 9800 > 5520. Easily.
  - DB: starts Feb 20. ~50 working days. 22 days at 170 = 3740, 28 days at 135 = 3780. Total = 7520 > 4035. Yes.
  - But we want exactly on-time. Let me use lower values to make it tight but achievable.
  - Actually, the task just says Yes, not barely. Let me use: 10hr window = 22 days starting around Feb 5. Web=160, DB=160 during window. Outside window after Feb 5: Web=135, DB=135 (but DB=0 before Feb 20).
  - Let me pick 22 consecutive working days starting Feb 5 for the window.

Now write the Python script. Here is the detailed script:

```python
import openpyxl
from openpyxl.utils import get_column_letter
from datetime import date, timedelta

wb = openpyxl.Workbook()

# Date setup
start_date = date(2018, 1, 22)
dates = [start_date + timedelta(days=i) for i in range(100)]
assert dates[-1] == date(2018, 5, 1), f"Last date is {dates[-1]}"

holidays = {date(2018, 2, 19), date(2018, 3, 30)}

def is_working_day(d):
    return d.weekday() < 5 and d not in holidays

# PO schedule
po_web = {date(2018,1,22): 1065, date(2018,2,1): 855, date(2018,2,15): 900,
          date(2018,3,1): 900, date(2018,4,2): 900, date(2018,5,1): 900}
po_db = {date(2018,1,22): 855, date(2018,2,1): 555, date(2018,2,15): 900,
         date(2018,3,1): 575, date(2018,4,2): 575, date(2018,5,1): 575}

sheet_names = ['Current Capacity and Racks', 'Relocated Network Equipment', '10 hr Shift Relocate Network Eq']

# Remove default sheet
default = wb.active
default.title = sheet_names[0]
wb.create_sheet(sheet_names[1])
wb.create_sheet(sheet_names[2])

# Identify working days for 10hr shift window (sheet 3)
# Pick 22 working days starting from Feb 5
shift_window_days = set()
count = 0
for d in dates:
    if d >= date(2018, 2, 5) and is_working_day(d) and count < 22:
        shift_window_days.add(d)
        count += 1

for si, sname in enumerate(sheet_names):
    ws = wb[sname]
    
    # Headers
    ws['C2'] = 'Rack-Mount Web Servers'
    ws['F2'] = 'Blade Database Servers'
    ws['I2'] = 'Network Appliances'
    
    headers = ['Planned Production', 'Purchase Orders Due', 'Cumulative Open Purchase Orders (EOD)',
               'Planned Production', 'Purchase Orders Due', 'Cumulative Open Purchase Orders (EOD)',
               'Actual Var to PO', 'Total Prod', 'Notes']
    for ci, h in enumerate(headers):
        ws.cell(row=3, column=3+ci, value=h)
    
    # Data rows
    for ri, d in enumerate(dates):
        row = ri + 4
        
        # Column B: date
        if ri == 0:
            ws.cell(row=row, column=2, value=d)
            ws.cell(row=row, column=2).number_format = 'YYYY-MM-DD'
        else:
            ws.cell(row=row, column=2).value = f'=B{row-1}+1'
        
        # PO due (cols D, G) - numeric constants
        d_web_po = po_web.get(d, 0)
        d_db_po = po_db.get(d, 0)
        ws.cell(row=row, column=4, value=d_web_po)
        ws.cell(row=row, column=7, value=d_db_po)
        
        # Determine planned production based on sheet
        wd = is_working_day(d)
        
        if si == 0:  # Current Capacity and Racks
            if not wd:
                web_prod, db_prod, net_prod = 0, 0, 0
            else:
                web_prod = 75
                # DB cannot start before Mar 1
                if d < date(2018, 3, 1):
                    db_prod = 0
                else:
                    db_prod = 60
                net_prod = 18
        
        elif si == 1:  # Relocated Network Equipment
            if not wd:
                web_prod, db_prod, net_prod = 0, 0, 0
            else:
                # Web
                if d < date(2018, 2, 5):
                    web_prod = 120
                else:
                    web_prod = 80
                # DB cannot start before Feb 20
                if d < date(2018, 2, 20):
                    db_prod = 0
                else:
                    db_prod = 60
                # Network: >=100 before Feb 1, 0 on/after Feb 1
                if d < date(2018, 2, 1):
                    net_prod = 13
                else:
                    net_prod = 0
        
        elif si == 2:  # 10 hr Shift Relocate Network Eq
            net_prod = 0  # always 0
            if not wd:
                web_prod, db_prod = 0, 0
            else:
                if d in shift_window_days:
                    web_prod = 160
                    # DB cannot start before Feb 20
                    if d < date(2018, 2, 20):
                        db_prod = 0
                    else:
                        db_prod = 160
                else:
                    if d < date(2018, 2, 5):
                        web_prod = 120
                    else:
                        web_prod = 135
                    if d < date(2018, 2, 20):
                        db_prod = 0
                    elif d < date(2018, 2, 5):
                        db_prod = 120
                    else:
                        db_prod = 135
        
        # Write planned production (cols C, F, I) - numeric constants
        ws.cell(row=row, column=3, value=web_prod)
        ws.cell(row=row, column=6, value=db_prod)
        ws.cell(row=row, column=9, value=net_prod)
        
        # Cumulative Open PO formulas (cols E, H)
        if ri == 0:
            ws.cell(row=row, column=5).value = f'=D{row}-C{row}'
            ws.cell(row=row, column=8).value = f'=G{row}-F{row}'
        else:
            ws.cell(row=row, column=5).value = f'=E{row-1}+D{row}-C{row}'
            ws.cell(row=row, column=8).value = f'=H{row-1}+G{row}-F{row}'
        
        # Total Prod formula (col J)
        ws.cell(row=row, column=10).value = f'=C{row}+F{row}+I{row}'

wb.save('/root/server_provisioning_recovery_plan_analysis.xlsx')
print('Workbook saved.')

# Verification
print('\nVerification:')
for si, sname in enumerate(sheet_names):
    ws = wb[sname]
    print(f'\nSheet: {sname}')
    print(f'  C2={ws["C2"].value}, F2={ws["F2"].value}, I2={ws["I2"].value}')
    print(f'  Row 3 headers: {[ws.cell(3,c).value for c in range(3,12)]}')
    print(f'  Row 4 B={ws.cell(4,2).value}, C={ws.cell(4,3).value}, D={ws.cell(4,4).value}')
    print(f'  Row 103 B={ws.cell(103,2).value}')
    
    # Sum productions
    total_web = sum(ws.cell(r,3).value for r in range(4,104))
    total_db = sum(ws.cell(r,6).value for r in range(4,104))
    total_net = sum(ws.cell(r,9).value for r in range(4,104))
    print(f'  Total Web={total_web}, DB={total_db}, Net={total_net}')
    print(f'  Total Web PO={sum(ws.cell(r,4).value for r in range(4,104))}')
    print(f'  Total DB PO={sum(ws.cell(r,7).value for r in range(4,104))}')
```

Run this script and check the output. Then verify:

1. **Sheet 1 (Current Capacity)**: Web total < 5520 (PO total) → May PO Not On-Time. DB total < 4035 → Not On-Time. Net total >= 1200.
2. **Sheet 2 (Relocated Network)**: Web total >= 5520 → On-Time. DB total < 4035 → Not On-Time. Net before Feb 1 >= 100, Net on/after Feb 1 = 0.
3. **Sheet 3 (10hr Shift)**: Web total >= 5520 → On-Time. DB total >= 4035 → On-Time. Net = 0. Shift window day count = 22 (between 20-24).

If any verification fails, adjust the production rates accordingly. Specifically check:
- Sheet 1: if Web total >= 5520, reduce web_prod (e.g., to 70). If Net total < 1200, increase net_prod.
- Sheet 2: if Web total < 5520, increase web_prod. If DB total >= 4035, reduce db_prod.
- Sheet 3: if totals don't meet PO, increase shift window days or rates.

Also verify that for Sheet 3, the count of days where Web > 135 OR Database > 135 is between 20 and 24. Note that some shift window days before Feb 20 will have DB=0 (not > 135), so only Web > 135 counts those days. Recount accordingly.

### Step 2: Create the summary markdown

After the workbook is verified, create `/root/server_provisioning_recovery_summary.md` with this exact content:

```markdown
# Server Provisioning Recovery Plan Summary

## Scenario 1

**Actions:** Maintain current rack capacity and provisioning rates. Web and Database server production operates within existing infrastructure limits. Network Appliance provisioning continues at standard pace throughout the recovery period.

**Rack-Mount Web Servers Impact:** Production limited to 75 units per working day. Total output falls short of cumulative purchase order demand of 5,520 units by May 1.

**Blade Database Servers Impact:** Production cannot begin until March 1 due to rack availability constraints. At 60 units per working day, total output is insufficient to meet cumulative purchase order demand of 4,035 units.

**Network Appliances Impact:** Steady production of 18 units per working day yields over 1,200 units across the recovery period, meeting minimum appliance requirements.

**May PO On-Time: No**

## Scenario 2

**Actions:** Relocate network equipment to free rack space earlier, enabling increased Web server throughput and an earlier Database server production start date of February 20. Network Appliance provisioning is front-loaded before February 1 and ceases afterward to free capacity.

**Rack-Mount Web Servers Impact:** Production at 120 units/day before February 5 and 80 units/day afterward accumulates enough output to fulfill all Web purchase orders on time by May 1.

**Blade Database Servers Impact:** Database production begins February 20 at 60 units per working day. Despite the earlier start compared to Scenario 1, cumulative output still falls short of the 4,035-unit purchase order total by May 1.

**Network Appliances Impact:** At least 100 Network Appliance units are completed before February 1. Production drops to zero from February 1 onward as relocated network equipment occupies former appliance provisioning space.

**May PO On-Time: Web Yes, Database No**

## Scenario 3

**Actions:** Implement temporary 10-hour shifts to increase daily Web and Database server throughput beyond normal limits. A 30-day notification is issued to operations staff before the extended shift period begins. Network Appliance production is suspended for the entire horizon to maximize server provisioning capacity.

**Rack-Mount Web Servers Impact:** During the temporary 10-hour shift window (22 working days starting February 5), Web production increases to 160 units/day. Combined with standard-rate days, total Web output exceeds the cumulative 5,520-unit purchase order requirement by May 1.

**Blade Database Servers Impact:** Database production begins February 20. During the 10-hour shift window, Database output reaches 160 units/day on eligible days. Total Database production surpasses the 4,035-unit cumulative purchase order target by May 1.

**Network Appliances Impact:** Network Appliance production is zero for the entire recovery period. All freed capacity is redirected to Web and Database server provisioning.

**May PO On-Time: Yes**
```

### Step 3: Final verification

Re-open the saved xlsx and verify:
1. Exactly 3 sheet names match exactly.
2. C2, F2, I2 values on each sheet.
3. C3:K3 header values on each sheet.
4. Row 4 date = 2018-01-22, Row 103 date formula resolves correctly.
5. Columns E, H, J contain formula strings (start with '='), columns C, D, F, G, I contain integers.
6. Weekend/holiday rows have C=F=I=0.
7. PO values on correct dates.
8. All scenario constraints are met.
9. The markdown file exists and contains all required sections and fields.

If any check fails, fix and re-verify before completing.

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

Task-local resources are available under `environment/skills`: Automotiveproductplanning, Directorofoperations, bc-calculated-fields-manufacturing, dispatching-parallel-agents, writing-plans.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=codex@example.com, author_name=Codex, category=datacenter-capacity-planning, difficulty=hard, tags=[excel, operations, capacity-planning, server-provisioning, datacenter].
Verifier config: timeout_sec=900.0.