"""Checks related to the .env file in the repository.

Usage:
    python src/scripts/fix_dot_env_file.py [--non-interactive] [--include-openai]
"""

import subprocess
from getpass import getpass
from pathlib import Path

import click

# List of all the environment variables that requested via UI
DESIRED_ENVIRONMENT_VARIABLES = dict(
    GPG_KEY_ID="Enter GPG key ID or leave empty if you do not want to use it. Type "
    "`gpg --list-secret-keys --keyid-format=long | grep sec | sed -E "
    "'s/.*\/([^ ]+).*/\\1/'` to see your key ID:\n> ",  # noqa
    GIT_NAME="Enter your full name, to be shown in Git commits:\n> ",
    GIT_EMAIL="Enter your email, as registered on your Github account:\n> ",
    POSTGRES_PASSWORD="Enter a PostgreSQL superuser password for timescale:\n> ",
    GF_SECURITY_ADMIN_PASSWORD="Enter a Grafana admins password:\n> ",
)

# List all predefined environment variables
PREDEFINED_ENVIRONMENT_VARIABLES = dict(
    GF_INSTALL_PLUGINS="volkovlabs-echarts-panel",
    POSTGRES_DB="{{cookiecutter.library_name}}_data",
    POSTGRES_USER="admin",
)

OPENAI_ENVIRONMENT_VARIABLES = dict(OPENAI_API_KEY="Enter your OpenAI API key:\n> ")


@click.command()
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="If set, the script will not ask for user input.",
)
@click.option(
    "--include-openai",
    is_flag=True,
    default=False,
    help="If set, the script will also ask for OpenAI environment variables.",
)
def fix_dot_env_file(non_interactive: bool, include_openai: bool) -> None:
    """Ensures that the .env file exists and contains all desired variables.

    Args:
        non_interactive:
            If set, the script will not ask for user input.
        include_openai:
            If set, the script will also ask for OpenAI environment variables.
    """
    env_path = Path(".env")
    name_and_email_path = Path(".name_and_email")

    # Ensure that the files exists
    env_path.touch(exist_ok=True)
    name_and_email_path.touch(exist_ok=True)

    # Extract all the lines in the files
    env_file_lines = env_path.read_text().splitlines(keepends=False)
    name_and_email_file_lines = name_and_email_path.read_text().splitlines(
        keepends=False
    )

    # Extract all the environment variables in the files
    env_vars = {line.split("=")[0]: line.split("=")[1] for line in env_file_lines}
    name_and_email_vars = {
        line.split("=")[0]: line.split("=")[1] for line in name_and_email_file_lines
    }

    desired_env_vars = DESIRED_ENVIRONMENT_VARIABLES
    if include_openai:
        desired_env_vars |= OPENAI_ENVIRONMENT_VARIABLES

    # For each of the desired environment variables, check if it exists in the .env
    # file
    env_vars_missing = [
        env_var for env_var in desired_env_vars.keys() if env_var not in env_vars
    ]

    # Create all the missing environment variables
    with env_path.open("a") as f:
        for env_key, env_var in PREDEFINED_ENVIRONMENT_VARIABLES.items():
            f.write(f'{env_key}="{env_var}"\n')

        for env_var in env_vars_missing:
            value = ""

            if env_var in name_and_email_vars:
                value = name_and_email_vars[env_var]
            elif env_var == "GPG_KEY_ID":
                gpg = subprocess.Popen(
                    ["gpg", "--list-secret-keys", "--keyid-format=long"],
                    stdout=subprocess.PIPE,
                )
                grep = subprocess.Popen(
                    ["grep", "sec"], stdin=gpg.stdout, stdout=subprocess.PIPE
                )
                value = (
                    subprocess.check_output(
                        ["sed", "-E", "s/.*\\/([^ ]+).*/\\1/"], stdin=grep.stdout
                    )
                    .decode()
                    .strip("\n")
                )
                gpg.wait()
                grep.wait()

            if value == "" and not non_interactive:
                if "PASSWORD" in env_var.upper():
                    value = getpass(desired_env_vars[env_var])
                else:
                    value = input(desired_env_vars[env_var])

            f.write(f"{env_var}={value}\n")

    # Remove the name and email file
    name_and_email_path.unlink()


if __name__ == "__main__":
    fix_dot_env_file()
