#!/bin/bash

set -e

# Make sure clone is up-to-date with remote.
git pull > /dev/null
git tag -l | xargs git tag -d > /dev/null
git fetch -t > /dev/null

TAG=$(python -c 'from olmo_core.version import VERSION; print("v" + VERSION)')

# Make sure tag/release doesn't already exist.
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://github.com/allenai/Olmo-core/releases/tag/${TAG}")
if [[ $STATUS_CODE == "200" ]]; then
    echo "Release tag ${TAG} already exists"
    exit 1
fi

python src/scripts/release/prepare_changelog.py

read -rp "Creating new release for $TAG. Do you want to continue? [Y/n] " prompt

if [[ $prompt == "y" || $prompt == "Y" || $prompt == "yes" || $prompt == "Yes" ]]; then
    git add -A
    git commit -m "(chore) prepare for release $TAG" || true && git push
    echo "Creating new git tag $TAG"
    git tag "$TAG" -m "$TAG"
    git push --tags
else
    echo "Canceled"
    git checkout CHANGELOG.md
    exit 1
fi
