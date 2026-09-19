"""
schemas_info.py - Official NebulaX Hackathon Problem Statement 3 (PS3) Specifications
Detailed schemas, data headers, scoring metrics, and LTA Smart Depot operational decision guidelines
for the 4 train subsystems.
"""

SUBSYSTEM_SCHEMAS = {
    "door": {
        "id": "door",
        "name": "Train Door Subsystem",
        "code": "PS3-SUB01-DOOR",
        "category": "Passenger Access & Actuation",
        "summary": "Temporal segment detection & abnormal resistance classification in continuous telemetry stream",
        "primary_metric": "IoU-weighted F1 Score",
        "scoring_formula": "Weighted F1 matching temporal start/end boundaries and 'Normal' vs 'Abnormal resistance' labels.",
        "input_formats": [".csv", ".xlsx"],
        "expected_output_filename": "door_predictions.csv",
        "output_columns": ["start_time", "end_time", "prediction"],
        "valid_predictions": ["Normal", "Abnormal resistance"],
        "depot_decision_impact": "High. Door jamming or obstruction failures account for >35% of mainline dwell delays. Abnormal motor current spikes indicate mechanical guide rail friction, worn rollers, or actuator degradation requiring immediate bay inspection before passenger entrapment risk occurs.",
        "headers": [
            {"parameter": "Datetime", "unit": "Timestamp", "type": "string", "description": "Format: YYYY-MM-DD-HH-MM-SS-MS (sampling time)"},
            {"parameter": "Car Type", "unit": "-", "type": "string", "description": "Motor car (M) or Trailer car (T)"},
            {"parameter": "Car Number", "unit": "-", "type": "integer", "description": "Car sequence number in 6 or 8-car trainset (01-08)"},
            {"parameter": "Door Number", "unit": "-", "type": "integer", "description": "Door leaf ID (1L, 1R, 2L, 2R, 3L, 3R, 4L, 4R)"},
            {"parameter": "Motor current (mA)", "unit": "mA", "type": "float", "description": "Drive motor current; abnormal friction causes severe current spikes"},
            {"parameter": "Motor Voltage (10mV)", "unit": "10mV", "type": "float", "description": "Drive motor supply voltage"},
            {"parameter": "Motor back electromotive force", "unit": "mV", "type": "float", "description": "Motor back-EMF proportional to instantaneous motor rotational speed"},
            {"parameter": "Door opening time (0.1 s)", "unit": "0.1s", "type": "float", "description": "Cumulative duration for complete door opening stroke"},
            {"parameter": "Door closing time (0.1 s)", "unit": "0.1s", "type": "float", "description": "Cumulative duration for complete door closing stroke"},
            {"parameter": "Close command", "unit": "binary (0/1)", "type": "integer", "description": "1 triggers door-closing action from Train Control System"},
            {"parameter": "Open command", "unit": "binary (0/1)", "type": "integer", "description": "1 triggers door-opening action from Train Control System"},
            {"parameter": "DCSR", "unit": "binary (0/1)", "type": "integer", "description": "Door Close Switch Right status (transition confirms mechanical closure)"},
            {"parameter": "DCSL", "unit": "binary (0/1)", "type": "integer", "description": "Door Close Switch Left status (transition confirms mechanical closure)"},
            {"parameter": "DLSR", "unit": "binary (0/1)", "type": "integer", "description": "Door Locked Switch Right status (confirms mechanical lock hook engaged)"},
            {"parameter": "DLSL", "unit": "binary (0/1)", "type": "integer", "description": "Door Locked Switch Left status (confirms mechanical lock hook engaged)"},
            {"parameter": "Door Opened", "unit": "binary (0/1)", "type": "integer", "description": "Indicator that door leaves reached fully opened position limit"},
            {"parameter": "Door Locked", "unit": "binary (0/1)", "type": "integer", "description": "Indicator that safety interlocking circuit is fully energized"},
            {"parameter": "Door is opening", "unit": "binary (0/1)", "type": "integer", "description": "Dynamic flag during active opening motion profile"},
            {"parameter": "Door is closing", "unit": "binary (0/1)", "type": "integer", "description": "Dynamic flag during active closing motion profile"},
            {"parameter": "Door leaf position", "unit": "mm", "type": "float", "description": "Linear position measurement along door header guide track"}
        ]
    },
    "acv": {
        "id": "acv",
        "name": "ACV (Air Conditioning & Ventilation)",
        "code": "PS3-SUB02-ACV",
        "category": "Thermal & Climate Control",
        "summary": "Refrigerant leakage fault diagnosis and localization across 8 train cars",
        "primary_metric": "Linear Rank-Decay Score",
        "scoring_formula": "Score = (n - (r - 1)) / n, where n = total cars (8), r = rank position of true leaking car.",
        "input_formats": [".xlsx", ".csv"],
        "expected_output_filename": "acv_predictions.csv",
        "output_columns": ["file_id", "ranked_cars"],
        "output_format_example": "03|01|05|02|04|06|07|08",
        "depot_decision_impact": "Critical. ACV failures in Singapore's tropical climate quickly lead to passenger discomfort and safety alerts. Refrigerant undercharge causes compressor overheating and emergency HVAC shutdown. Identifies precisely which car needs bay shunting for leak testing and R407C/R134a refrigerant recharge.",
        "headers": [
            {"parameter": "Car model", "unit": "-", "type": "string", "description": "Rolling stock family (e.g. C751B, C151B, C830, R151)"},
            {"parameter": "Train number", "unit": "-", "type": "string", "description": "Trainset identifier (e.g. 0408, 315)"},
            {"parameter": "Time", "unit": "Timestamp", "type": "datetime", "description": "Multivariate telemetry sampled every 30 seconds"},
            {"parameter": "Car <NN> - ACV Setting Mode", "unit": "Mode ID", "type": "integer", "description": "Operating setting: Auto / Ventilation / Cooling / Pre-cool"},
            {"parameter": "Car <NN> - ACV Running Mode", "unit": "State ID", "type": "integer", "description": "Actual HVAC stage: Off, Full Cool (100%), Half Cool (50%), Vent"},
            {"parameter": "Car <NN> - Control Temperature (Cooling)", "unit": "°C", "type": "float", "description": "Thermostat setpoint target for cooling mode"},
            {"parameter": "Car <NN> - Control Temperature (Heating)", "unit": "°C", "type": "float", "description": "Thermostat setpoint target for heating/dehumidification"},
            {"parameter": "Car <NN> - Indoor Average Temperature", "unit": "°C", "type": "float", "description": "Averaged saloon interior temperature sensor readings across car"},
            {"parameter": "Car <NN> - Outside Temperature Sensor Reading", "unit": "°C", "type": "float", "description": "External ambient air temperature measured at roof intake"},
            {"parameter": "Car <NN> - Load Halved", "unit": "binary (0/1)", "type": "integer", "description": "Compressor capacity staging flag (1 = running at 50% capacity)"},
            {"parameter": "Car <NN> - ACV Information Valid", "unit": "binary (0/1)", "type": "integer", "description": "Communication status flag with HVAC microprocessor unit"}
        ]
    },
    "rail": {
        "id": "rail",
        "name": "Rail Corrugation Subsystem",
        "code": "PS3-SUB03-RAIL",
        "category": "Wheel-Rail Interface & Track Safety",
        "summary": "3-class classification (Normal / Side I / Side II) from 64-channel axle-box vibration and shock",
        "primary_metric": "Macro F1 Score",
        "scoring_formula": "Unweighted arithmetic average of F1 scores across Normal, Side I, and Side II classes.",
        "input_formats": [".csv", ".xlsx"],
        "expected_output_filename": "rail_predictions.csv",
        "output_columns": ["file_id", "prediction"],
        "valid_predictions": ["Normal", "Side I", "Side II"],
        "depot_decision_impact": "High. Rail corrugation accelerates bogie component fatigue, causes excessive track noise, and increases derailment risk. Pinpoints whether the train running gear or track section on Side I vs Side II is inducing periodic contact resonance, triggering rail grinding or wheel lathe reprofiling.",
        "headers": [
            {"parameter": "Rotational speed (Column 1)", "unit": "pulses", "type": "binary pulse", "description": "Speed sensor on toothed wheel (90 teeth, wheel diameter 0.85m); pulse frequency calculates instantaneous train velocity"},
            {"parameter": "Car 1..8, Position 1, 3, 5, 7 Vibration & Shock", "unit": "m/s²", "type": "float", "description": "Side I vertical axle-box acceleration and shock signals (10,000 Hz, 1-second window)"},
            {"parameter": "Car 1..8, Position 2, 4, 6, 8 Vibration & Shock", "unit": "m/s²", "type": "float", "description": "Side II vertical axle-box acceleration and shock signals (10,000 Hz, 1-second window)"},
            {"parameter": "Total Sensor Channels", "unit": "Channels", "type": "count", "description": "128 acceleration channels (64 vibration + 64 shock) across all 8 cars of the train"}
        ]
    },
    "shm": {
        "id": "shm",
        "name": "SHM (Structural Health Monitoring)",
        "code": "PS3-SUB04-SHM",
        "category": "Bogie & Carbody Structural Integrity",
        "summary": "Dynamic stress time series regression for cumulative fatigue damage assessment",
        "primary_metric": "max(0, 1 - MAPE)",
        "scoring_formula": "Score = max(0, 1 - MAPE), where MAPE = mean(|true - pred| / |true|). Perfect = 1.0; floors at 0 if MAPE >= 100%.",
        "input_formats": [".csv", ".xlsx"],
        "expected_output_filename": "shm_predictions.csv",
        "output_columns": ["file_id", "prediction"],
        "depot_decision_impact": "Critical. Rail vehicle bogie frames and carbody bolsters sustain cyclic alternating loads. Cumulative damage estimation via rainflow counting and Miner's rule prevents catastrophic weld fatigue cracks. Determines when a trainset must be pulled for ultrasonic NDT testing or structural reinforcement.",
        "headers": [
            {"parameter": "Dynamic Stress Time Series", "unit": "MPa", "type": "float time-series", "description": "Continuous dynamic stress measurements recorded from strain gauges mounted on critical structural weld points"},
            {"parameter": "Load Condition Context", "unit": "AW Class", "type": "metadata", "description": "Operational load state: AW0 (empty tare weight) and AW4 (full passenger crush load)"},
            {"parameter": "Target Output Value", "unit": "Damage (D)", "type": "float (0.0 to 1.0+)", "description": "Cumulative fatigue damage index based on Miner's Rule (D = sum(n_i / N_i)). D >= 1.0 denotes fatigue limit failure"}
        ]
    }
}
