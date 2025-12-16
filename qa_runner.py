import re
import subprocess
import os

CHECKLIST_FILE = "docs/qa/checklist_source.md"
REPORT_FILE = "docs/qa/falsification_report.md"

def main():
    with open(CHECKLIST_FILE, "r") as f:
        content = f.read()

    # Regex to find tests: **N. Description** ... ```bash command ```
    # Note: re.DOTALL is needed for multi-line match, but we need to be careful not to consume too much.
    # We'll split by "**" which seems to separate tests.
    
    tests = []
    
    # Simple state machine parser
    lines = content.split('\n')
    current_test_num = None
    current_desc = None
    current_command = None
    in_code_block = False
    
    for line in lines:
        # Match test header: **1. Verify ...**
        m_header = re.match(r'\*\*(\d+)\.\s+(.*?)\*\*', line)
        if m_header:
            current_test_num = int(m_header.group(1))
            current_desc = m_header.group(2)
            current_command = []
            continue

        if line.strip().startswith("```bash"):
            in_code_block = True
            continue
        
        if line.strip().startswith("```") and in_code_block:
            in_code_block = False
            if current_test_num is not None:
                cmd = "\n".join(current_command).strip()
                tests.append({
                    "num": current_test_num,
                    "desc": current_desc,
                    "cmd": cmd
                })
                current_test_num = None # Reset
            continue

        if in_code_block:
            if current_command is not None:
                current_command.append(line)

    print(f"Found {len(tests)} tests.")
    
    results = []
    passed = 0
    failed = 0
    blocked = 0

    # Ensure environment variables are set
    # export DEMO_DIR="demos/realtime-transcription"
    env = os.environ.copy()
    env["DEMO_DIR"] = "demos/realtime-transcription"

    print("Running tests...")
    
    for test in tests:
        print(f"Running Test {test['num']}: {test['desc']}")
        
        try:
            # We run the command using bash explicitly
            # Capture stdout and stderr
            process = subprocess.run(
                ['bash', '-c', test['cmd']], 
                env=env, 
                capture_output=True, 
                text=True,
                cwd="/home/noah/src/whisper.apr" # Enforce project root
            )
            
            output = process.stdout.strip()
            # Also check stderr if stdout is empty, but usually echo goes to stdout
            if not output and process.stderr:
                output = process.stderr.strip()
                
            status = "UNKNOWN"
            if "SURVIVED" in output:
                status = "PASS"
                passed += 1
            elif "FALSIFIED" in output:
                status = "FAIL"
                failed += 1
            else:
                # Fallback logic if grep fails or command errors out without echoing
                if process.returncode != 0:
                     status = "FAIL (Command Error)"
                     failed += 1
                else:
                     status = "BLOCKED (No Output)"
                     blocked += 1
            
            results.append({
                "test": test,
                "status": status,
                "output": output
            })
            
        except Exception as e:
            results.append({
                "test": test,
                "status": "BLOCKED",
                "output": str(e)
            })
            blocked += 1

    # Generate Report
    with open(REPORT_FILE, "w") as f:
        f.write("# WAPR-SPEC-010 Falsification Report\n\n")
        f.write(f"**Date:** {subprocess.check_output(['date']).decode().strip()}\n")
        f.write(f"**Total Tests:** {len(tests)}\n")
        f.write(f"**Passed (Survived):** {passed}\n")
        f.write(f"**Failed (Falsified):** {failed}\n")
        f.write(f"**Blocked:** {blocked}\n\n")
        
        f.write("| ID | Description | Status | Output |\n")
        f.write("|----|-------------|--------|--------|\n")
        for res in results:
            # Escape pipes in output for markdown table
            safe_output = res['output'].replace("|", "\\|").replace("\n", " ")
            f.write(f"| {res['test']['num']} | {res['test']['desc']} | **{res['status']}** | `{safe_output}` |\n")

    print(f"Report generated at {REPORT_FILE}")

if __name__ == "__main__":
    main()
