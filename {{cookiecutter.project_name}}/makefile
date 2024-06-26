# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: docs help

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Create .env file if it does not already exist
ifeq (,$(wildcard .env))
  $(shell touch .env)
endif
{{'\n# Create poetry env file if it does not already exist\nifeq (,$(wildcard ${HOME}/.poetry/env))\n  $(shell mkdir ${HOME}/.poetry)\n  $(shell touch ${HOME}/.poetry/env)\nendif\n' if cookiecutter.dependency_manager != 'pip'}}
# Includes environment variables from the .env file
include .env

# Set gRPC environment variables, which prevents some errors with the `grpcio` package
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

# Ensure that `pipx`{{' and `poetry`' if cookiecutter.dependency_manager != 'pip'}} will be able to run, since `pip` and `brew` put these
# in the following folders on Unix systems
export PATH := ${HOME}/.local/bin:/opt/homebrew/bin:$(PATH)
{{'\n# Prevent DBusErrorResponse during `poetry install`\n#(see https://stackoverflow.com/a/75098703 for more information)\nexport PYTHON_KEYRING_BACKEND := keyring.backends.null.Keyring\n' if cookiecutter.dependency_manager != 'pip'}}
help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "Installing the '{{ cookiecutter.project_name }}' project..."
	@$(MAKE) --quiet install-brew
	@$(MAKE) --quiet install-gpg
	@$(MAKE) --quiet generate-gpg-key{{'\n	@$(MAKE) --quiet install-pipx\n	@$(MAKE) --quiet install-poetry' if cookiecutter.dependency_manager != 'pip'}}
	@$(MAKE) --quiet install-dependencies
	@$(MAKE) --quiet setup-environment-variables
	@$(MAKE) --quiet setup-git{{'\n\t@$(MAKE) --quiet freeze' if cookiecutter.dependency_manager == 'pip'}}
	@$(MAKE) --quiet add-repo-to-git
	@echo "Installed the '{{ cookiecutter.project_name }}' project. You can now activate your virtual environment with 'source .venv/bin/activate'."
	@echo "Note that this is a {{'Poetry' if cookiecutter.dependency_manager != 'pip' else 'pip'}} project. Use '{{'poetry add <package>' if cookiecutter.dependency_manager != 'pip' else 'pip install <package>'}}' to install new dependencies and '{{'poetry remove <package>' if cookiecutter.dependency_manager != 'pip' else 'pip uninstall <package>'}}' to remove them.{{' Freeze your dependencies into pyproject.toml with \'make freeze\'.' if cookiecutter.dependency_manager == 'pip'}}"

install-brew:
	@if [ $$(uname) = "Darwin" ] && [ "$(shell which brew)" = "" ]; then \
		/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; \
		echo "Installed Homebrew."; \
	fi

install-gpg:
	@if [ "$(shell which gpg)" = "" ] || [ "$(shell which gpg-agent)" = "" ]; then \
		uname=$$(uname); \
			case $${uname} in \
				(*Linux*) distro=$$(lsb_release -i | sed 's/Distributor ID:\t//'); \
					case $${distro} in \
						(*Ubuntu*) installCmd='apt-get update && apt-get install -y gnupg gpg-agent'; ;; \
						(*CentOS*) installCmd='yum install -y gnupg gpg-agent'; ;; \
						(*) echo 'Could not automatically install gnupg for the $${distro} distribution. Please manually install gnupg and try again.'; exit 2; ;; \
					esac; ;; \
				(*Darwin*) installCmd='brew install gnupg pinentry-mac'; ;; \
				(*) echo 'Could not automatically install gnupg. Please manually install gnupg and try again.'; exit 2; ;; \
			esac; \
			"$${installCmd}"; \
		echo "Installed gnupg."; \
	fi

generate-gpg-key:
	@if [ "$(shell gpg --list-secret-keys --keyid-format=long | grep sec | sed -E 's/.*\/([^ ]+).*/\1/')" = "" ]; then \
		echo "Generating a new GPG key - please follow the prompts."; \
		gpg --full-generate-key; \
		echo "Generated a new GPG key. Remember to register it to Github at https://github.com/settings/gpg/new, where you add the key generated by running 'gpg --armor --export <key>'"; \
	fi

install-pipx:
	@if [ "$(shell which pipx)" = "" ]; then \
		uname=$$(uname); \
			case $${uname} in \
				(*Darwin*) installCmd='brew install pipx'; ;; \
				(*CYGWIN*) installCmd='py -3 -m pip install --upgrade --user pipx'; ;; \
				(*) installCmd='python3 -m pip install --upgrade --user pipx'; ;; \
			esac; \
			$${installCmd}; \
		pipx ensurepath --force; \
		echo "Installed pipx."; \
	fi
{{'\ninstall-poetry:\n\t@if [ ! "$(shell poetry --version)" = "Poetry (version 1.8.2)" ]; then \\\n        python3 -m pip uninstall -y poetry poetry-core poetry-plugin-export; \\\n        pipx install --force poetry==1.8.2; \\\n        echo "Installed Poetry."; \\\n    fi\n' if cookiecutter.dependency_manager != 'pip'}}
install-dependencies:
	{{'@poetry env use python' if cookiecutter.dependency_manager != 'pip' else '@python'}}{{cookiecutter.python_version}}{{' && poetry install' if cookiecutter.dependency_manager != 'pip' else ' -m venv .venv\n\t@. .venv/bin/activate && pip install -qU pip && pip install -qe .[dev]'}}

setup-environment-variables:
	@if [ -d src/ragger ]; then \
		{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}python src/scripts/fix_dot_env_file.py --include-openai; \
	else \
		{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}python src/scripts/fix_dot_env_file.py; \
	fi

setup-environment-variables-non-interactive:
	@if [ -d src/ragger ]; then \
		{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}python src/scripts/fix_dot_env_file.py --include-openai --non-interactive; \
	else \
		{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}python src/scripts/fix_dot_env_file.py --non-interactive; \
	fi

setup-git:
	@git config --global init.defaultBranch main
	@git init
	@git config --local user.name ${GIT_NAME}
	@git config --local user.email ${GIT_EMAIL}
	@if [ ${GPG_KEY_ID} = "" ]; then \
		echo "No GPG key ID specified. Skipping GPG signing."; \
		git config --local commit.gpgsign false; \
	else \
		git config --local commit.gpgsign true; \
		git config --local user.signingkey ${GPG_KEY_ID}; \
		echo "Signed with GPG key ID ${GPG_KEY_ID}."; \
	fi
	@{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}pre-commit install

add-repo-to-git:
	@export GPG_TTY=$(tty)
	@gpgconf --kill gpg-agent
	@if [ ! "$(shell git status --short)" = "" ] && [ "$(shell git --no-pager log --all | sed 's/`//g')" = "" ]; then \
		git add .; \
		git commit --quiet -m "Initial commit"; \
	fi
	@if [ "$(shell git remote)" = "" ]; then \
		git remote add origin git@github.com:alexandrainst/{{ cookiecutter.project_name }}.git; \
	fi

{{'\nfreeze:  ## Freeze dependencies\n\t@. .venv/bin/activate && python src/scripts/freeze_dependencies.py\n\t@echo "Updated dependencies in pyproject.toml."\n' if cookiecutter.dependency_manager == 'pip'}}

docs:  ## Generate documentation
	@{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}pdoc --docformat google src/{{cookiecutter.project_name}} -o docs
	@echo "Saved documentation."

view-docs:  ## View documentation
	@echo "Viewing API documentation..."
	@uname=$$(uname); \
		case $${uname} in \
			(*Linux*) openCmd='xdg-open'; ;; \
			(*Darwin*) openCmd='open'; ;; \
			(*CYGWIN*) openCmd='cygstart'; ;; \
			(*) echo 'Error: Unsupported platform: $${uname}'; exit 2; ;; \
		esac; \
		"$${openCmd}" docs/{{ cookiecutter.project_name }}.html

test:  ## Run tests
	@{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}pytest && {{'poetry run ' if cookiecutter.dependency_manager != 'pip'}}readme-cov

tree:  ## Print directory tree
	@tree -a --gitignore -I .git .

setup-ragger:
	@if [ -d src/ragger ]; then \
		echo "RAG functionality already added to the project. Skipping."; \
		exit 1; \
	fi
	@if [ ! -d .venv ]; then \
		echo "The project hasn't been installed yet. Please run 'make install' first."; \
		exit 1; \
	fi
	@echo "Adding RAG functionality to the project..."
	@cd src && git clone git@github.com:alexandrainst/ragger.git && rm -rf ragger/.git && cd -
	@mv src/ragger/src/scripts/run_demo.py src/scripts/
	@mv src/ragger/src/scripts/run_cli.py src/scripts/
	@cp -R src/ragger/config/* config/
	@if [ ! -f data/processed/document_store.jsonl ]; then \
		mv src/ragger/data/processed/document_store.jsonl data/processed/; \
	fi
	@{{'poetry add' if cookiecutter.dependency_manager != 'pip' else 'pip install'}} -e src/ragger
	@{{'poetry run ' if cookiecutter.dependency_manager != 'pip' else '. .venv/bin/activate && '}}python src/scripts/fix_dot_env_file.py --include-openai

add-rag:  ## Add RAG functionality to the project
	@$(MAKE) --quiet setup-ragger
	@git add .
	@echo "Added RAG functionality to the project, but haven't committed the changes yet. Please commit them manually."

update-rag:  ## Update the RAG submodule - this requires `make add-rag` to have been run
	@if [ -d src/ragger ]; then \
		rm -rf src/ragger; \
		$(MAKE) --quiet setup-ragger; \
		git add .; \
		echo "Updated RAG functionality, but haven't committed the changes yet (if any). Please commit them manually."; \
	else \
		echo "RAG code not found. Please run 'make add-rag' first."; \
	fi
