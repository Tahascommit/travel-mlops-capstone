# Jenkins with the tools our pipeline needs:
#   docker cli + buildx  -> builds images on docker desktop through the mounted docker socket
#   kubectl              -> deploys to the docker desktop kubernetes cluster
#   uv                   -> creates the python 3.12 env for tests / model export
FROM jenkins/jenkins:2.580.1-lts-jdk21

USER root
ARG TARGETARCH
ARG DOCKER_VERSION=29.9.2
ARG BUILDX_VERSION=v0.38.1
ARG KUBECTL_VERSION=v1.36.1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN ARCH=$([ "$TARGETARCH" = "arm64" ] && echo aarch64 || echo x86_64) \
 && curl -fsSL --http1.1 --retry 4 --retry-all-errors "https://download.docker.com/linux/static/stable/${ARCH}/docker-${DOCKER_VERSION}.tgz" \
    | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
 && mkdir -p /usr/local/lib/docker/cli-plugins \
 && curl -fsSL --http1.1 --retry 4 --retry-all-errors -o /usr/local/lib/docker/cli-plugins/docker-buildx \
    "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-${TARGETARCH}" \
 && chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx \
 && curl -fsSL --http1.1 --retry 4 --retry-all-errors -o /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
 && chmod +x /usr/local/bin/kubectl

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

USER jenkins

# plugins are installed at build time so the container starts ready to use
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
