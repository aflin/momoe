#!/bin/sh
# Update Momoe without starting a model: the newest code (git pull, when this folder is a git clone), then what a start
# would refresh - the engines compiled again where their source changed (engine/, and momoe-metal / momoe-cuda in
# build-metal/ / build-cuda/), the Python packages, each installed model's settings and draft subset. The model files are
# not touched. Start a model later with ./momoe start NAME. Options are passed on to setup.py.
cd "$(dirname "$0")" || exit 1
# all of it in a function, read before it runs: the git pull below can change this very file
main() {
  if [ -e .git ]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "This folder is a git clone, but git is not installed: install it (sudo apt install git, or on a Mac"
      echo "xcode-select --install) or run \"git pull\" here yourself, then run ./update.sh again."
      exit 1
    fi
    echo "Getting the newest Momoe (git pull) ..."
    if ! git pull --ff-only; then
      echo
      echo "git pull did not succeed (the reason is above): nothing was updated. Files you changed here can stop it:"
      echo "\"git status\" lists them."
      exit 1
    fi
  else
    echo "This copy of Momoe was not made with git, so it cannot fetch new files itself. Download the newest one:"
    echo "  https://github.com/aflin/momoe/archive/refs/heads/main.zip"
    echo "unzip it anywhere and run ./momoe setup in it: it finds the model files this copy uses (its data folder is"
    echo "recorded in ~/.config/momoe) and sets itself up the same way - nothing big is downloaded again."
    echo "Checking this copy's engines and settings meanwhile ..."
  fi
  exec sh ./setup.sh --update "$@"
}
main "$@"
