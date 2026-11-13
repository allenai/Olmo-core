#!/bin/bash

set -e

repo_url=https://github.com/allenai/Olmo-core

tags=$(git tag -l --sort=-version:refname 'v*' | head -n 2)
current_tag=$(echo "$tags" | head -n 1)
last_tag=$(echo "$tags" | tail -n 1)

echo "Current release: $current_tag"
echo "Last release: $last_tag"

if [ -z "$last_tag" ]; then
    echo "No previous release, nothing to do"
    exit 0;
fi

commits_since_last_release=$(git log "${last_tag}..${current_tag}" --format=format:%H)

echo "Commits/PRs since last release:"
for commit in $commits_since_last_release; do
    pr_number=$(gh pr list --search "$commit" --state merged --json number --jq '.[-1].number')
    if [ -z "$pr_number" ]; then
        echo "$commit"
    else
        echo "$commit (PR #$pr_number)"
        gh pr comment "$pr_number" --body "This PR has been released in [${current_tag}](${repo_url}/releases/tag/${current_tag})."
    fi
done
