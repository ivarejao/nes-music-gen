#!/bin/bash

source .env
apptainer exec \
  --env DATA_MOUNT_POINT=$DATA_MOUNT_POINT \
  --bind $DATA_MOUNT_POINT:/data \
  --pwd /app \
  nesmdb.sif \
  gunicorn -w 4 -b 0.0.0.0:5555 --timeout 3600 app:app

