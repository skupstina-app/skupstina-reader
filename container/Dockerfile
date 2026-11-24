# Speaker reader: yt-dlp (+ Deno for YouTube's JS challenges), PyAV, Tesseract with Serbian models.
FROM denoland/deno:bin AS deno

FROM python:3.13-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-srp tesseract-ocr-srp-latn ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=deno /deno /usr/local/bin/deno
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY reader ./reader
ENV PYTHONUNBUFFERED=1 OMP_THREAD_LIMIT=1
EXPOSE 8080
CMD ["python", "-m", "reader.main"]
