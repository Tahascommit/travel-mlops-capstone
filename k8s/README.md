# Kubernetes

Manifests for running the API on the Kubernetes cluster that comes with Docker Desktop.

| File | What |
|---|---|
| `namespace.yml` | `travel-mlops` namespace |
| `deployment.yml` | the API pods: readiness/liveness probes on `/health`, cpu/memory requests and limits, rolling updates with `maxUnavailable: 0`, and a `preStop` sleep so pods don't drop requests while shutting down |
| `service.yml` | LoadBalancer service, on Docker Desktop it is reachable on http://localhost:8000 |
| `hpa.yml` | HorizontalPodAutoscaler, 2 to 8 pods, target 50% cpu |
| `load_test.py` | sends batch predictions from many threads to watch the autoscaler |

The deployment has no `replicas` field on purpose, the HPA decides how many pods run.

## Deploying by hand

```bash
docker build -t flight-price-api:latest .
kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/
kubectl -n travel-mlops rollout status deployment/flight-price-api
kubectl -n travel-mlops get pods,svc,hpa
```

`imagePullPolicy: IfNotPresent` is used because the image only exists locally, Docker Desktop's cluster can use images built with the local docker. Normally Jenkins does the deploy and sets the image tag to the build number.

## metrics-server (needed for the HPA)

Docker Desktop doesn't ship metrics-server. Without it the HPA shows `<unknown>` as the cpu target.

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.9.0/components.yaml
kubectl -n kube-system patch deployment metrics-server --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl top pods -n travel-mlops
```

The `--kubelet-insecure-tls` flag is needed on Docker Desktop because the kubelet certificate has no IP address in it, so metrics-server can't verify it. It is fine for a local cluster but should not be used on a real one.

## Watching it scale

```bash
python k8s/load_test.py --minutes 4 --workers 16
kubectl -n travel-mlops get hpa -w          # in a second terminal
```

## Useful commands

```bash
kubectl -n travel-mlops rollout history deployment/flight-price-api    # one revision per jenkins build
kubectl -n travel-mlops rollout undo deployment/flight-price-api       # go back one version
kubectl -n travel-mlops logs -l app=flight-price-api --tail=20
kubectl -n travel-mlops describe hpa flight-price-api
```
