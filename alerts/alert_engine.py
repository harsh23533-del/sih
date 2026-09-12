"""
Week 7 — Early Warning Engine (alert generation stage).

Implements the last stage of the plan's pipeline: data update -> preprocessing
-> AI prediction -> risk score -> threshold check -> warning generation.
This module does the threshold check and message generation; in the
prototype the alert surfaces on the dashboard, a production version would
push it via email/SMS/API (stubbed here with a delivery log, per the plan).

Requirements:
    pip install pandas
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Alert fires at High/Critical by default — tune per district in consultation
# with disaster-management authorities, per the plan's False-alarm tolerance metric.
ALERT_LEVELS = {"High", "Critical"}


@dataclass
class Alert:
    location: str
    risk_score: float
    risk_level: str
    system_action: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def check_threshold(risk_level: str, alert_levels: set = ALERT_LEVELS) -> bool:
    """Returns True if this risk level should trigger an alert."""
    return risk_level in alert_levels


def generate_message(location: str, risk_score: float, risk_level: str, system_action: str) -> str:
    """Builds the human-readable warning message shown on the dashboard /
    sent to disaster-management contacts."""
    return (
        f"[{risk_level.upper()} RISK] {location}: landslide risk score {risk_score:.0f}/100. "
        f"Recommended action: {system_action}. This is an AI-based decision-support signal, "
        f"not a guarantee of occurrence — validate with local conditions before acting."
    )


def process_alert(location: str, risk_score: float, risk_level: str, system_action: str,
                   alert_levels: set = ALERT_LEVELS) -> Alert | None:
    """Runs the threshold check and, if crossed, returns an Alert with its
    generated message. Returns None if below the alert threshold."""
    if not check_threshold(risk_level, alert_levels):
        return None
    message = generate_message(location, risk_score, risk_level, system_action)
    return Alert(location, risk_score, risk_level, system_action, message)


def log_delivery(alert: Alert, channel: str = "dashboard", log_path: str = "alert_delivery_log.csv") -> None:
    """Appends a delivery record. Prototype only logs to the dashboard
    (channel='dashboard'); swap in real email/SMS/API calls here for
    production and keep logging every attempt for accountability, per the
    plan's Early Warning Engine note."""
    import csv
    import os

    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "location", "risk_level", "risk_score", "channel", "message"])
        writer.writerow([alert.timestamp, alert.location, alert.risk_level,
                          alert.risk_score, channel, alert.message])


if __name__ == "__main__":
    # Quick sanity check
    alert = process_alert("Gangtok, Sikkim", 87, "Critical", "Immediate alert / authority review")
    if alert:
        print(alert.message)
        log_delivery(alert)
        print("Logged to alert_delivery_log.csv")
