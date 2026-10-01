IMAGE=inspectron
APP_HOST=inspectron
WEB_HOST=localhost:7000

.DEFAULT_GOAL := help
.PHONY: help


help:
	@echo "--------------------------------------------"
	@printf "\033[37m%10s  \033[0m%s\n" App     $(APP_HOST)
	@printf "\033[37m%10s  \033[0m%s\n" Web     $(WEB_HOST)
	@printf "\033[37m%10s  \033[0m%s\n" Image   $(IMAGE)
	@echo "--------------------------------------------"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'





build: ## build image
	docker compose build

start: ## start app
	docker compose up -d

stop: ## stop app
	docker compose down --volumes

enter: ## enter shell in app
	docker exec -it $(WEB_HOST) bash


log: ## show container log
	docker logs $(APP_HOST)  -f
