import os
import sys
import json
import time
import datetime
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# FAULT CLASSIFICATION
# ─────────────────────────────────────────────

AI_RESOLVABLE = {
    "5": "Transient Path Clear (Self-Healed)",
}

HUMAN_REQUIRED = {
    "1": "Heavy Short Circuit Fault",
    "2": "SF6 Gas Leakage Anomaly",
    "3": "Lethal Broken Conductor Loop",
    "4": "Transformer Thermal Overload",
}

FAULT_PRIORITY = {
    "3": 1,  # Most critical
    "1": 2,
    "2": 3,
    "4": 4,
    "5": 5,  # Least critical
}


def classify_response(fault_label: int) -> str:
    if str(fault_label) in AI_RESOLVABLE:
        return "AI_RESOLVABLE"
    elif str(fault_label) in HUMAN_REQUIRED:
        return "HUMAN_REQUIRED"
    return "UNKNOWN"


# ─────────────────────────────────────────────
# TOOL FUNCTIONS
# ─────────────────────────────────────────────

def read_grid_log(zone_id: str, timestamp: str = None) -> dict:
    try:
        df = pd.read_csv("data/grid_logs.csv")
        zone_data = df[df["ZoneID"] == zone_id]
        if timestamp:
            row_data = zone_data[zone_data["Timestamp"] == timestamp]
            if not row_data.empty:
                row = row_data.iloc[0]
            else:
                row = zone_data.tail(1).iloc[0]
        else:
            row = zone_data.tail(1).iloc[0]
        return {
            "zone_id": zone_id,
            "voltage_v": row["Voltage"],
            "current_a": row["Current"],
            "gas_pressure": row["GasPressure"],
            "temperature": row["Temperature"],
            "fault_label": int(row["FaultLabel"]),
            "timestamp": row["Timestamp"]
        }
    except Exception as e:
        return {"error": str(e)}


def lookup_fault_signature(fault_label: int) -> dict:
    try:
        with open("knowledge_base/fault_signatures.json", "r") as f:
            data = json.load(f)
        signatures = data["signatures"]
        key = str(fault_label)
        if key in signatures:
            fault = signatures[key]
            severity = "critical" if fault["confidence_score"] >= 95 else "major" if fault["confidence_score"] >= 90 else "minor"
            return {
                "fault_type": fault["type"],
                "confidence": fault["confidence_score"],
                "action": fault["recommended_action"],
                "safety_checklist": fault["safety_checklist"],
                "severity": severity
            }
        return {"fault_type": "unknown", "confidence": 0}
    except Exception as e:
        return {"error": str(e)}


# FIX #8: Zone topology now reads from JSON file instead of hardcoded dict,
# with a built-in fallback so the agent never crashes if the file is missing.
_FALLBACK_TOPOLOGY = {
    "Zone_1": {"connected_zones": ["Zone_2", "Zone_3"], "breaker_id": "BRK-001", "assigned_crew": "Team Alpha"},
    "Zone_2": {"connected_zones": ["Zone_1", "Zone_4"], "breaker_id": "BRK-002", "assigned_crew": "Team Beta"},
    "Zone_3": {"connected_zones": ["Zone_1", "Zone_5"], "breaker_id": "BRK-003", "assigned_crew": "Team Alpha"},
    "Zone_4": {"connected_zones": ["Zone_2", "Zone_6"], "breaker_id": "BRK-004", "assigned_crew": "Team Gamma"},
    "Zone_5": {"connected_zones": ["Zone_3", "Zone_6"], "breaker_id": "BRK-005", "assigned_crew": "Team Beta"},
    "Zone_6": {"connected_zones": ["Zone_4", "Zone_5"], "breaker_id": "BRK-006", "assigned_crew": "Team Gamma"},
}

def check_zone_topology(zone_id: str) -> dict:
    try:
        with open("data/zone_topology.json", "r") as f:
            data = json.load(f)
        zones = data["zones"]
    except Exception:
        zones = _FALLBACK_TOPOLOGY

    if zone_id in zones:
        zone_data = zones[zone_id]
        return {
            "connected_zones": zone_data.get("connected_to", []),
            "breaker_id": zone_data.get("breaker_id", "UNKNOWN"),
            "assigned_crew": zone_data.get("assigned_crew", "Unassigned")
        }
    return {"error": f"Zone {zone_id} not found in topology"}

def generate_dispatch(zone_id: str, fault_type: str,
                      severity: str, confidence: float,
                      action: str, checklist: list,
                      crew: str, breaker_id: str,
                      readings: dict, response_time: float) -> str:
    dispatch = f"""
╔══════════════════════════════════════════════════╗
           GRIDMIND EMERGENCY DISPATCH
╚══════════════════════════════════════════════════╝

🔴 FAULT TYPE     : {fault_type.upper()}
📍 ZONE           : {zone_id}
⚠️  SEVERITY       : {severity.upper()}
🎯 CONFIDENCE     : {confidence}%
👷 ASSIGNED CREW  : {crew}
🔌 BREAKER ID     : {breaker_id}
⏱️  RESPONSE TIME  : {response_time:.2f} seconds

📊 LIVE READINGS:
   Voltage      : {readings['voltage_v']} kV
   Current      : {readings['current_a']} A
   Gas Pressure : {readings['gas_pressure']} bar
   Temperature  : {readings['temperature']} °C
   Timestamp    : {readings['timestamp']}

📋 RECOMMENDED ACTION:
   {action}

✅ SAFETY CHECKLIST:
"""
    for i, item in enumerate(checklist, 1):
        dispatch += f"   {i}. {item}\n"
    dispatch += "\n⚡ STATUS: DISPATCHED AUTOMATICALLY BY GRIDMIND\n"
    return dispatch


# ─────────────────────────────────────────────
# CORE AGENT LOOP
# ─────────────────────────────────────────────

def run_gridmind_agent(zone_id: str, timestamp: str = None):
    steps = []
    start_time = datetime.datetime.now()

    # Step 1: Read grid data
    steps.append(f"🔍 Reading grid data for {zone_id}...")
    log = read_grid_log(zone_id, timestamp)

    if "error" in log:
        steps.append(f"❌ Error reading data: {log['error']}")
        return steps, None, 0

    steps.append(f"📊 Readings — Voltage: {log['voltage_v']}kV | "
                 f"Current: {log['current_a']}A | "
                 f"Gas Pressure: {log['gas_pressure']}bar | "
                 f"Temp: {log['temperature']}°C")

    # Step 2: Check fault label
    steps.append("🧠 Analyzing fault label from sensor data...")
    fault_label = log["fault_label"]

    if fault_label == 0:
        steps.append("✅ Fault Label 0 — Normal operation. No anomalies detected.")
        return steps, None, 0

    steps.append(f"⚡ Fault Label {fault_label} detected — initiating diagnosis...")

    # Step 3: Look up fault signature
    steps.append("📚 Cross-referencing fault knowledge base...")
    diagnosis = lookup_fault_signature(fault_label)

    if "error" in diagnosis:
        steps.append(f"❌ Knowledge base error: {diagnosis['error']}")
        return steps, None, 0

    if diagnosis["fault_type"] == "unknown":
        steps.append("⚠️  Unknown fault pattern — flagging for human review.")
        return steps, "HUMAN_REVIEW", 0

    steps.append(f"🎯 Diagnosis: {diagnosis['fault_type']} "
                 f"(Confidence: {diagnosis['confidence']}%)")

    # Step 4: Check confidence threshold
    if diagnosis["confidence"] < 70:
        steps.append(f"⚠️  Confidence too low ({diagnosis['confidence']}%) "
                     f"— flagging for human review.")
        return steps, "HUMAN_REVIEW", 0

    # Step 5: Check zone topology
    steps.append(f"🗺️  Checking zone topology for {zone_id}...")
    topology = check_zone_topology(zone_id)

    if "error" in topology:
        steps.append(f"❌ Topology error: {topology['error']}")
        return steps, None, 0

    steps.append(f"🔌 Breaker: {topology['breaker_id']} | "
                 f"Crew: {topology['assigned_crew']} | "
                 f"Connected zones: {', '.join(topology['connected_zones'])}")

    # Step 6: Groq AI verification
    steps.append("🤖 Consulting Groq AI for final verification...")
    prompt = f"""
You are GridMind, an autonomous power grid fault diagnosis AI agent.

Zone: {zone_id}
Readings: Voltage={log['voltage_v']}kV, Current={log['current_a']}A,
Gas Pressure={log['gas_pressure']}bar, Temperature={log['temperature']}°C
Fault Label: {fault_label}
Preliminary Diagnosis: {diagnosis['fault_type']} (Confidence: {diagnosis['confidence']}%)
Recommended Action: {diagnosis['action']}

In 2-3 sentences, confirm this diagnosis and explain why these readings indicate this fault type.
Be concise and technical.
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        ai_reasoning = response.choices[0].message.content.strip()
        steps.append(f"💡 AI Verification: {ai_reasoning}")
    except Exception as e:
        steps.append(f"⚠️  Groq verification skipped: {str(e)}")

    # Step 7: Generate dispatch
    end_time = datetime.datetime.now()
    response_time = (end_time - start_time).total_seconds()

    steps.append(f"🚨 Confidence {diagnosis['confidence']}% — "
                 f"Isolating {zone_id} and generating dispatch...")

    dispatch = generate_dispatch(
        zone_id=zone_id,
        fault_type=diagnosis["fault_type"],
        severity=diagnosis["severity"],
        confidence=diagnosis["confidence"],
        action=diagnosis["action"],
        checklist=diagnosis["safety_checklist"],
        crew=topology["assigned_crew"],
        breaker_id=topology["breaker_id"],
        readings=log,
        response_time=response_time
    )

    steps.append(f"✅ Dispatch generated in {response_time:.2f}s — sent to response team.")
    return steps, dispatch, response_time


# ─────────────────────────────────────────────
# FAULT RESOLUTION TRACKER
# ─────────────────────────────────────────────

def load_resolved_sites() -> set:
    """Load list of sites already resolved from tracker file."""
    try:
        with open("reports/resolved_sites.json", "r") as f:
            data = json.load(f)
            return set(data["resolved"])
    except:
        return set()


def mark_site_resolved(zone_id: str, fault_type: str, resolved_by: str):
    """Mark a site as resolved in the tracker."""
    os.makedirs("reports", exist_ok=True)
    resolved = load_resolved_sites()
    resolved.add(zone_id)

    try:
        with open("reports/resolved_sites.json", "r") as f:
            data = json.load(f)
    except:
        data = {"resolved": [], "history": []}

    data["resolved"] = list(resolved)
    data["history"].append({
        "zone": zone_id,
        "fault_type": fault_type,
        "resolved_by": resolved_by,
        "resolved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    with open("reports/resolved_sites.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ {zone_id} marked as RESOLVED by {resolved_by}")


def show_resolution_history():
    """Show all previously resolved sites."""
    try:
        with open("reports/resolved_sites.json", "r") as f:
            data = json.load(f)
        if not data["history"]:
            print("No resolved sites yet.")
            return
        print("\n📋 RESOLUTION HISTORY:")
        print("-"*60)
        for item in data["history"]:
            print(f"   ✅ {item['zone']} | {item['fault_type']} | "
                  f"Resolved by: {item['resolved_by']} | {item['resolved_at']}")
    except:
        print("No resolution history found.")


def reset_resolved_sites():
    """Clear all resolved sites — start fresh."""
    if os.path.exists("reports/resolved_sites.json"):
        os.remove("reports/resolved_sites.json")
    if os.path.exists("reports/resolved_rows.json"):
        os.remove("reports/resolved_rows.json")
    print("🔄 Resolution tracker cleared — all sites active again.")


if __name__ == "__main__":

    # ── HANDLE COMMAND LINE ARGUMENTS ──
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--resolve":
            if len(sys.argv) < 4:
                print("Usage: python agent/gridmind_agent.py --resolve <ZoneID> <TeamName>")
            else:
                zone = sys.argv[2]
                resolved_by = sys.argv[3]
                df = pd.read_csv("data/grid_logs.csv")
                zone_row = df[df["ZoneID"] == zone]
                if not zone_row.empty:
                    faulty_zone = zone_row[zone_row["FaultLabel"] != 0]
                    if not faulty_zone.empty:
                        fault_label = int(faulty_zone["FaultLabel"].iloc[0])
                        fault_name = HUMAN_REQUIRED.get(str(fault_label),
                                     AI_RESOLVABLE.get(str(fault_label), "Unknown"))
                    else:
                        fault_name = "Unknown"
                else:
                    fault_name = "Unknown"
                mark_site_resolved(zone, fault_name, resolved_by)
                print(f"\n🔁 Run 'python agent/gridmind_agent.py' to see remaining faults.")
            sys.exit(0)

        elif command == "--history":
            show_resolution_history()
            sys.exit(0)

        elif command == "--reset":
            reset_resolved_sites()
            sys.exit(0)

    # FIX #6: Input validation — fail gracefully instead of crashing mid-demo
    if not os.path.exists("data/zone_topology.json"):
        print("⚠️  WARNING: data/zone_topology.json not found — using fallback topology.")

    if not os.path.exists("knowledge_base/fault_signatures.json"):
        print("❌ ERROR: knowledge_base/fault_signatures.json not found. Cannot diagnose faults.")
        sys.exit(1)

    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY not found in .env — AI verification will be skipped for all faults.\n")

    # ── LOAD DATA ──
    df = pd.read_csv("data/grid_logs.csv")
    all_faulty = df[df["FaultLabel"] != 0].copy()
    normal_count = len(df[df["FaultLabel"] == 0])

    # Load resolved rows tracker
    os.makedirs("reports", exist_ok=True)
    resolved_rows_file = "reports/resolved_rows.json"
    try:
        with open(resolved_rows_file, "r") as f:
            resolved_data = json.load(f)
            resolved_indices = set(resolved_data["resolved_indices"])
    except:
        resolved_indices = set()

    # Filter out already resolved rows
    remaining_faulty = all_faulty[~all_faulty.index.isin(resolved_indices)].copy()
    remaining_faulty["priority"] = remaining_faulty["FaultLabel"].astype(str).map(FAULT_PRIORITY)
    remaining_faulty = remaining_faulty.sort_values(["priority", "Timestamp"]).reset_index()

    print("\n" + "="*60)
    print("       ⚡ GRIDMIND AUTONOMOUS MONITORING SYSTEM ⚡")
    print("="*60)
    print(f"📊 Total Readings Scanned     : {len(df)}")
    print(f"✅ Normal Readings             : {normal_count}")
    print(f"🚨 Total Faulty Rows           : {len(all_faulty)}")
    print(f"✅ Already Resolved            : {len(resolved_indices)} faults")
    print(f"🔴 Remaining Faults            : {len(remaining_faulty)} faults")
    print(f"📍 Affected Zones              : {remaining_faulty['ZoneID'].nunique()}")
    print("="*60)

    if remaining_faulty.empty:
        print(f"\n🎉 ALL {len(all_faulty)} FAULTS RESOLVED — Grid operating normally!")
        print("Run '--reset' to start a new monitoring cycle.")
        show_resolution_history()
        sys.exit(0)

    print("\n🔍 Scanning all remaining faults by priority...\n")
    time.sleep(0.5)

    # ── TRACKING ──
    ai_resolved_list = []
    crew_dispatched_list = []
    human_review_list = []
    new_resolved_indices = set()

    # FIX #3: Topology-aware team assignment, restored.
    # Each zone has a "home" crew from zone_topology.json. We only fall back
    # to round-robin if a zone has no assigned crew on record.
    from collections import deque
    TEAM_QUEUE = deque(["Team Alpha", "Team Beta", "Team Gamma"])
    TEAM_STATS = {"Team Alpha": {"jobs": 0}, "Team Beta": {"jobs": 0}, "Team Gamma": {"jobs": 0}}

    def get_fallback_team():
        team = TEAM_QUEUE[0]
        TEAM_QUEUE.rotate(-1)
        TEAM_STATS[team]["jobs"] += 1
        return team

    # ── PROCESS ALL REMAINING FAULTS ──
    for _, row in remaining_faulty.iterrows():
        zone_id = row["ZoneID"]
        fault_label = int(row["FaultLabel"])
        timestamp = str(row["Timestamp"])
        original_index = row["index"]
        response_type = classify_response(fault_label)
        fault_name = HUMAN_REQUIRED.get(str(fault_label),
                     AI_RESOLVABLE.get(str(fault_label), "Unknown"))

        print(f"\n{'='*60}")
        print(f"🚨 FAULT #{len(ai_resolved_list) + len(crew_dispatched_list) + len(human_review_list) + 1} DETECTED")
        print(f"   📍 Zone      : {zone_id}")
        print(f"   🕐 Time      : {timestamp}")
        print(f"   ⚠️  Fault     : {fault_name}")
        print(f"   📊 Priority  : P{FAULT_PRIORITY.get(str(fault_label), 99)}")
        print(f"{'='*60}")
        time.sleep(0.2)

        # Run agent
        reasoning_steps, result, response_time = run_gridmind_agent(zone_id, timestamp)

        print("\n── AGENT REASONING TRACE ──")
        for step in reasoning_steps:
            print(f"   {step}")
            time.sleep(0.1)

        print("\n── GRIDMIND DECISION ──")

        # ── AI RESOLVABLE ──
        if response_type == "AI_RESOLVABLE":
            block = f"""
┌─────────────────────────────────────────────────────┐
│  🤖 AI AUTONOMOUS RESOLUTION                        │
├─────────────────────────────────────────────────────┤
│  Zone      : {zone_id:<40}│
│  Fault     : {fault_name:<40}│
│  Time      : {timestamp:<40}│
│  Action    : Auto-recloser sequence executed        │
│  Resp Time : {response_time:.2f}s                                  │
│  Status    : ✅ RESOLVED BY GRIDMIND AI             │
└─────────────────────────────────────────────────────┘"""
            print(block)
            ai_resolved_list.append({
                "zone": zone_id,
                "fault": fault_name,
                "timestamp": timestamp,
                "response_time": response_time
            })
            new_resolved_indices.add(original_index)
            mark_site_resolved(zone_id, fault_name, "GridMind AI")

        # ── HUMAN REQUIRED ──
        elif response_type == "HUMAN_REQUIRED":
            if result and result != "HUMAN_REVIEW":
                print(result)

            # FIX #3: prefer the zone's home crew from topology;
            # fall back to round-robin only if topology has no crew listed.
            topology = check_zone_topology(zone_id)
            team = topology.get("assigned_crew") if "error" not in topology else None
            if not team:
                team = get_fallback_team()
            else:
                TEAM_STATS.setdefault(team, {"jobs": 0})
                TEAM_STATS[team]["jobs"] += 1

            block = f"""
┌─────────────────────────────────────────────────────┐
│  🚒 CREW DISPATCHED                                 │
├─────────────────────────────────────────────────────┤
│  Zone      : {zone_id:<40}│
│  Fault     : {fault_name:<40}│
│  Time      : {timestamp:<40}│
│  Team      : {team:<40}│
│  Action    : Field crew dispatched to site          │
│  Resp Time : {response_time:.2f}s                                  │
│  Status    : 🔴 TEAM EN ROUTE                       │
└─────────────────────────────────────────────────────┘"""
            print(block)
            crew_dispatched_list.append({
                "zone": zone_id,
                "fault": fault_name,
                "timestamp": timestamp,
                "team": team,
                "response_time": response_time
            })
            new_resolved_indices.add(original_index)

        # ── UNKNOWN ──
        else:
            print(f"⚠️  Unclassified fault — flagged for human review.")
            human_review_list.append({
                "zone": zone_id,
                "fault": "Unknown",
                "timestamp": timestamp
            })
            new_resolved_indices.add(original_index)

        time.sleep(0.3)

    # ── SAVE RESOLVED ROWS ──
    all_resolved = resolved_indices | new_resolved_indices
    with open(resolved_rows_file, "w") as f:
        json.dump({"resolved_indices": list(all_resolved)}, f)

    # ── FINAL REPORT ──
    total_processed = len(ai_resolved_list) + len(crew_dispatched_list) + len(human_review_list)
    total_remaining = len(all_faulty) - len(all_resolved)

    summary = f"""
{'='*60}
       📋 GRIDMIND MISSION REPORT
       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

📊 OVERALL GRID STATUS:
   Total Fault Rows in Dataset : {len(all_faulty)}
   Processed This Run          : {total_processed}
   Cumulative Resolved         : {len(all_resolved)}
   Still Pending               : {max(total_remaining, 0)}

📊 RESOLUTION BREAKDOWN (THIS RUN):
   🤖 Resolved by AI autonomously : {len(ai_resolved_list)} faults
   🚒 Dispatched to field crews   : {len(crew_dispatched_list)} faults
   ⚠️  Flagged for human review    : {len(human_review_list)} faults
   ──────────────────────────────────────────────────
   📍 Total processed this run    : {total_processed} faults

🤖 AI RESOLVED ({len(ai_resolved_list)} faults):
"""
    if ai_resolved_list:
        for item in ai_resolved_list:
            summary += f"   ✅ {item['zone']} | {item['fault']} | {item['timestamp']} | {item['response_time']:.2f}s\n"
    else:
        summary += "   None this run\n"

    summary += f"\n🚒 CREW DISPATCHED ({len(crew_dispatched_list)} faults):\n"
    if crew_dispatched_list:
        for item in crew_dispatched_list:
            summary += f"   🔴 {item['zone']} | {item['fault']} | {item['timestamp']} | {item['team']}\n"
    else:
        summary += "   None this run\n"

    summary += f"\n⚠️  HUMAN REVIEW ({len(human_review_list)} faults):\n"
    if human_review_list:
        for item in human_review_list:
            summary += f"   ⚠️  {item['zone']} | {item['fault']} | {item['timestamp']}\n"
    else:
        summary += "   None\n"

    if total_remaining <= 0:
        summary += f"\n🎉 ALL {len(all_faulty)} FAULTS RESOLVED — Grid operating normally!\n"
    else:
        summary += f"\n⏳ {max(total_remaining, 0)} faults still pending — run again after crews report back.\n"

    summary += f"{'='*60}\n"

    print(summary)

    # Save report
    report_filename = f"reports/mission_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write("GRIDMIND MISSION REPORT\n")
        f.write(summary)
    print(f"📄 Report saved to: {report_filename}")