# Ollama meetup demo (5mn)

- Make sure that Docker is installed and running on your laptop with Kubernetes enabled.

- Install ollama, start it and pull the mistral model: 

    ```shell
    ollama pull mistral
    ```

    By default, the server is reachable on port 11434.

- Install the Python requirements in a virtualenv and activate it.

- To prove that you are not scamming people, start some random deployment in a custom namespace:

    ```shell
    kubectl create ns demo
    kubectl apply -f https://k8s.io/examples/controllers/nginx-deployment.yaml  -n demo
    ```

- Profit:

    ```shell
    python run.py
    ```

There is a lot to improve, namely:
- handling empty outputs gracefully
- asking explicitly for the namespace when it is not provided by the user
- just shelling out kubectl instead of installing the K8S Python client

...but that's the beauty of the game: giving ideas to the audience for building cool stuff !

