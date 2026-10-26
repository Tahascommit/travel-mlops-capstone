// CI/CD pipeline for the flight price API
//
// Runs on every new commit (jenkins polls github) and when the airflow retraining dag
// promotes a new champion model.
//
//   python env -> unit tests -> get model -> api tests -> build image -> deploy -> smoke test
//
// The image tag is the build number, so every deployment can be traced back to a build
// (and the model version it contains is shown on /model-info).

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    parameters {
        booleanParam(name: 'RETRAIN', defaultValue: false,
                     description: 'Train a new model in this build instead of using the champion from MLflow')
    }

    environment {
        IMAGE = 'flight-price-api'
        TAG = "${env.BUILD_NUMBER}"
        NAMESPACE = 'travel-mlops'
        MLFLOW_TRACKING_URI = 'http://mlflow:5001'
        MLFLOW_DISABLE_AGENT_HINT = '1'
        KUBECONFIG = "${env.WORKSPACE}/.kubeconfig"
        API_URL = 'http://host.docker.internal:8000'
    }

    stages {
        stage('Python env') {
            steps {
                sh '''
                    uv venv --python 3.12 --clear .venv
                    uv pip install --python .venv/bin/python -r api/requirements.txt -r requirements-train.txt pytest==9.1.1
                '''
            }
        }

        stage('Unit tests') {
            steps {
                sh '.venv/bin/pytest -q tests/test_features.py --junitxml=reports/unit-tests.xml'
            }
        }

        stage('Get model') {
            steps {
                script {
                    if (params.RETRAIN) {
                        sh '.venv/bin/python -m src.flight_price.train'
                    }
                    // either way we ship the champion, so a retrained model that wasn't good enough never gets deployed
                    sh '.venv/bin/python -m src.flight_price.export_champion'
                }
            }
        }

        stage('API tests') {
            steps {
                sh '.venv/bin/pytest -q tests/test_api.py --junitxml=reports/api-tests.xml'
            }
        }

        stage('Build image') {
            steps {
                sh 'docker build -t $IMAGE:$TAG -t $IMAGE:latest .'
                sh 'docker images $IMAGE --format "{{.Repository}}:{{.Tag}}  {{.Size}}" | head -5'
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                sh 'sh jenkins/kubeconfig.sh'
                sh '''
                    kubectl apply -f k8s/namespace.yml
                    kubectl apply -f k8s/service.yml -f k8s/hpa.yml
                    # same deployment file, only the image tag changes to this build
                    sed "s|image: $IMAGE:latest|image: $IMAGE:$TAG|" k8s/deployment.yml | kubectl apply -f -
                    kubectl -n $NAMESPACE annotate deployment/flight-price-api \
                        kubernetes.io/change-cause="jenkins build $TAG" --overwrite
                '''
                script {
                    try {
                        sh 'kubectl -n $NAMESPACE rollout status deployment/flight-price-api --timeout=180s'
                    } catch (err) {
                        // new pods never got healthy, go back to the previous version
                        sh 'kubectl -n $NAMESPACE rollout undo deployment/flight-price-api'
                        error("rollout failed, rolled back: ${err}")
                    }
                }
                sh 'kubectl -n $NAMESPACE get pods -o wide'
            }
        }

        stage('Smoke test') {
            steps {
                sh '''
                    for i in $(seq 1 20); do curl -sf $API_URL/health && break; sleep 3; done
                    curl -sf $API_URL/model-info
                    echo
                    curl -sf -X POST $API_URL/predict -H "Content-Type: application/json" \
                        -d '{"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass", "agency": "FlyingDrops"}'
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'reports/*.xml'
            // the kubeconfig has cluster credentials in it, don't leave it in the workspace
            sh 'rm -f .kubeconfig'
        }
        success {
            echo "deployed ${IMAGE}:${TAG}"
        }
    }
}
