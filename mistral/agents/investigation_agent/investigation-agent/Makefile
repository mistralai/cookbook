.PHONY: check format lint typecheck build serve send-error ask

check: lint typecheck
	env -u AGENT SENTRY_CONNECTOR_NAME=sentry_example GITHUB_CONNECTOR_NAME=github_example uv run agents info agent

format:
	uv run ruff format .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run ty check

build:
	docker build --platform linux/amd64 -t investigation-agent-runtime agent
	docker build --platform linux/amd64 -f workflow/Dockerfile -t investigation-agent-workflow .

# Runs the agent locally under INVESTIGATION_AGENT_NAME so it never collides with the deployed agent's name.
# DEPLOYMENT_NAME is the local workflow worker's queue, so it is unset here and the control plane picks the agent's queue.
serve:
	cd agent && uv run --env-file ../.env sh -c 'unset DEPLOYMENT_NAME; exec agents serve . --name "$${INVESTIGATION_AGENT_NAME:?Set INVESTIGATION_AGENT_NAME in .env}"'

# Sends the sample-app ValueError to the Sentry project in SENTRY_DSN.
send-error:
	cd sample-app && uv run --env-file ../.env python checkout.py

# Asks the agent `make serve` is running one question: make ask Q="your question". Without Q it asks about the sample-app error.
ask:
	cd agent && uv run --env-file ../.env python ask.py "$$Q"
