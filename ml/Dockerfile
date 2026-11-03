FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY ml/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY ml /app
RUN useradd --create-home --uid 10001 hospitalml && mkdir -p /data/datasets /data/models /data/updates && chown -R hospitalml:hospitalml /app /data
USER hospitalml
ENV LOCAL_DATA_ROOT=/data/datasets LOCAL_MODEL_ROOT=/data/models LOCAL_UPDATE_ROOT=/data/updates
VOLUME ["/data"]
EXPOSE 8000
CMD ["python", "service.py"]
