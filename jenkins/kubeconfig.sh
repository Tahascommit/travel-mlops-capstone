#!/bin/sh
# Makes a kubeconfig that works from inside the jenkins container.
#
# The kubeconfig on the mac points at https://127.0.0.1:<port>, but inside a container 127.0.0.1 is
# the container itself. So we point it at host.docker.internal instead, and tell kubectl to check the
# server certificate against the name "localhost" (which is in the api server cert). That way tls
# verification stays on.
set -e
: "${KUBECONFIG:?KUBECONFIG must be set}"

kubectl --kubeconfig /kube/config config view --raw --minify --context docker-desktop > "$KUBECONFIG"
CLUSTER=$(kubectl config view -o jsonpath='{.clusters[0].name}')
SERVER=$(kubectl config view -o jsonpath='{.clusters[0].cluster.server}')
kubectl config set-cluster "$CLUSTER" \
  --server="$(echo "$SERVER" | sed 's/127\.0\.0\.1/host.docker.internal/')" \
  --tls-server-name=localhost > /dev/null
kubectl get nodes
