.PHONY: setup dev test eval demo-reset docker

setup:        ## install all dependencies and seed the database
	./scripts/setup.sh

dev:          ## run backend, AI service and frontend locally
	./scripts/dev.sh

test:         ## backend unit + API tests, AI service offline tests, frontend lint + build
	cd backend && npm test && npm run test:e2e
	cd ai-service && LLM_PROVIDER=fake .venv/bin/python -m pytest
	cd frontend && npm run lint && npm run build

eval:         ## scenario evaluation against the running stack (real LLM)
	cd backend && npm run db:demo-reset
	ai-service/.venv/bin/python evals/run_evals.py
	cd backend && npm run db:demo-reset

demo-reset:   ## remove bookings created during demos, keep seed data
	cd backend && npm run db:demo-reset

docker:       ## full stack in containers (UI on :8080)
	docker compose up --build
