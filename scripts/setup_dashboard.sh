#!/bin/bash

# Function to install required packages
install_packages() {
    sudo apt-get update
    sudo apt-get install -y curl docker.io docker-compose
}

# Function to setup Redash
setup_redash() {
    mkdir -p ~/redash
    cd ~/redash

    cat <<EOF > docker-compose.yml
version: '3.7'
services:
  server:
    image: redash/redash:latest
    ports:
      - "5000:5000"
    environment:
      REDASH_DATABASE_URL: "postgresql://postgres:password@postgres/postgres"
      REDASH_REDIS_URL: "redis://redis:6379/0"
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:12-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password

  redis:
    image: redis:5.0-alpine
EOF

    sudo docker-compose up -d
    echo "Redash setup completed. Access it at http://localhost:5000"
}

# Function to setup Metabase
setup_metabase() {
    mkdir -p ~/metabase
    cd ~/metabase

    cat <<EOF > docker-compose.yml
version: '3.7'
services:
  metabase:
    image: metabase/metabase:latest
    ports:
      - "3000:3000"
    environment:
      MB_DB_TYPE: postgres
      MB_DB_DBNAME: metabase
      MB_DB_PORT: 5432
      MB_DB_USER: metabase
      MB_DB_PASS: password
      MB_DB_HOST: postgres

  postgres:
    image: postgres:12-alpine
    environment:
      POSTGRES_USER: metabase
      POSTGRES_PASSWORD: password
      POSTGRES_DB: metabase
EOF

    sudo docker-compose up -d
    echo "Metabase setup completed. Access it at http://localhost:3000"
}

# Function to setup Superset
setup_superset() {
    mkdir -p ~/superset
    cd ~/superset

    cat <<EOF > docker-compose.yml
version: '3.7'
services:
  superset:
    image: apache/superset:latest
    ports:
      - "8088:8088"
    environment:
      SUPERSET_SECRET_KEY: 'thisISaSECRET_1234'
    depends_on:
      - postgres
      - redis
    command: >
      /bin/sh -c "
      superset db upgrade &&
      superset init
      "

  postgres:
    image: postgres:12-alpine
    environment:
      POSTGRES_DB: superset
      POSTGRES_USER: superset
      POSTGRES_PASSWORD: password

  redis:
    image: redis:latest

  superset-init:
    image: apache/superset:latest
    depends_on:
      - superset
    command: /bin/sh -c "/app/docker/docker-init.sh"
EOF

    sudo docker-compose up -d
    echo "Superset setup completed. Access it at http://localhost:8088"
}

# Main script
if [ $# -ne 1 ]; then
    echo "Usage: $0 {redash|metabase|superset}"
    exit 1
fi

install_packages

case $1 in
    redash)
        setup_redash
        ;;
    metabase)
        setup_metabase
        ;;
    superset)
        setup_superset
        ;;
    *)
        echo "Invalid option. Usage: $0 {redash|metabase|superset}"
        exit 1
        ;;
esac
