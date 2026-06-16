import pandas as pd
import json
import os

def read_grid_log(zone_id):
    """
    Function 1: Intercepts and parses the latest telemetry metrics vector 
    from data/grid_logs.csv for the requested zone.
    """
    csv_path = 'data/grid_logs.csv'
    if not os.path.exists(csv_path):
        return {"error": "Dataset logs file path not found."}
        
    df = pd.read_csv(csv_path)
    zone_data = df[df['ZoneID'] == zone_id]
    
    if zone_data.empty:
        return {"Voltage": 11.0, "Current": 120.0, "GasPressure": 5.5, "Temperature": 65.0, "FaultLabel": 0}
        
    latest_row = zone_data.iloc[-1]
    return {
        "Voltage": float(latest_row['Voltage']),
        "Current": float(latest_row['Current']),
        "GasPressure": float(latest_row['GasPressure']),
        "Temperature": float(latest_row['Temperature']),
        "FaultLabel": int(latest_row['FaultLabel'])
    }

def lookup_fault_signature(fault_label):
    """
    Function 2: Connects with knowledge_base/fault_signatures.json 
    to retrieve expert engineering system instructions guidelines.
    """
    json_path = 'knowledge_base/fault_signatures.json'
    if not os.path.exists(json_path):
        return {"error": "Fault knowledge dictionary matrix missing."}
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    signatures_map = data.get("signatures", {})
    return signatures_map.get(str(fault_label), {
        "type": "Unknown Diagnostic Signature",
        "confidence_score": 0,
        "recommended_action": "Flag anomaly instantly for specialized human review.",
        "safety_checklist": ["Halt any automated tool tasks"]
    })

def check_zone_topology(zone_id):
    """
    Function 3: Reads spatial layouts routing maps from data/zone_topology.json 
    to locate grid connections and assigned lines maintenance crews.
    """
    json_path = 'data/zone_topology.json'
    if not os.path.exists(json_path):
        return {"error": "Topological zone configuration registry missing."}
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    return data.get("zones", {}).get(zone_id, {"error": f"Zone identifier token [{zone_id}] invalid."})