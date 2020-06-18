#!/bin/sh

PORT=9090
exec uvicorn airbnb:app --host 0.0.0.0 --port $PORT