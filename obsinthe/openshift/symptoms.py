import pandas as pd


def generate_alert_symptom_id_cluster_operator(a):
    parts = []
    parts.append(a["alertname"])

    if "labels" in a:
        labels = a["labels"]
    else:
        # make it work with both pagerduty and telemetry alerts
        labels = a

    if "name" in labels:
        parts.append(labels["name"])
    if "reason" in labels and not pd.isna(labels["reason"]):
        reason = labels["reason"][:42]  # shorten long strings
        parts.append(f"{reason}")
    return "|".join(p for p in parts if not pd.isna(p))


def generate_alert_symptom_id_cluster_generic(a):
    parts = []
    if a["namespace"]:
        parts.append(a["namespace"])
    parts.append(a["alertname"])

    if "labels" in a:
        labels = a["labels"]
    else:
        # make it work with both pagerduty and telemetry alerts
        labels = a

    if "poddisruptionbudget" in labels:
        parts.append(labels["poddisruptionbudget"])

    ret = "|".join(p for p in parts if not pd.isna(p))

    if "kubernetes_operator_part_of" in a and not pd.isna(
        labels["kubernetes_operator_part_of"]
    ):
        kube_parts = [a["kubernetes_operator_part_of"]]
        if "kubernetes_operator_component" in a and not pd.isna(
            labels["kubernetes_operator_component"]
        ):
            kube_parts.append(a["kubernetes_operator_component"])
        ret += ":" + "|".join(kube_parts)
    return ret


def generate_symptom_id(a):
    if pd.isna(a["alertname"]):
        return None
    if a["alertname"].startswith("ClusterOperator"):
        return generate_alert_symptom_id_cluster_operator(a)
    else:
        return generate_alert_symptom_id_cluster_generic(a)
