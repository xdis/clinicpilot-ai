FROM node:22-slim
RUN apt-get update -y && apt-get install -y --no-install-recommends openssl && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/package*.json ./
RUN npm ci
COPY backend/ ./
COPY mock-data/ /app/mock-data/
RUN npx prisma generate && npm run build
ENV DATABASE_URL=file:/data/clinic.db PORT=3000 MOCK_DATA_DIR=/app/mock-data
# non-root (the image's built-in "node" user, uid 1000); /data holds the SQLite database
RUN mkdir -p /data && chown node /data
USER node
EXPOSE 3000
# Migrate and (idempotently) seed on start, then serve.
CMD ["sh", "-c", "npx prisma migrate deploy && npx ts-node prisma/seed.ts && node dist/main.js"]
