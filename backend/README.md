# ClinicPilot backend

NestJS 11 + Prisma 6 + SQLite. Owns all data and every business rule; the AI service never touches the database directly.

```bash
cp .env.example .env
npm install
npm run db:setup      # migrate + seed from ../mock-data (idempotent)
npm run db:demo-reset # remove bookings created during demos, keep seed data
npm run start:dev     # http://localhost:3000/api, Swagger at /api/docs
npm test              # unit tests (booking rules)
npm run test:e2e      # API tests against a fresh prisma/test.db
```

See the root `README.md` and `TECH_DOC.md` for the full system.
