from kubernetes import client, config

def list_pods(namespace):
    # Load kubeconfig
    config.load_kube_config()
    # Create API client instance
    api_instance = client.CoreV1Api()
    # Call API to list all pods in the given namespace
    response = api_instance.list_namespaced_pod(namespace)
    out = {"pods": []}
    for pod in response.items:
        out["pods"].append({"name": pod.metadata.name, "status": pod.status.phase})
    return out


