#!/bin/bash

# Exit immediately if any command fails
set -e

git branch

# Input variables
echo "Enter branch name:"
read BRANCH_NAME

echo "Enter commit message:"
read COMMIT_MESSAGE

echo "Enter PR title:"
read PR_TITLE

echo "Enter PR description (optional, press Enter to skip):"
read PR_DESCRIPTION

# Checkout to the main branch and pull latest changes

# Check if the branch already exists
if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  echo "Branch '$BRANCH_NAME' already exists. Exiting..."

  echo "Updating the base branch..."
    git checkout "$BRANCH_NAME"
    git pull origin main

else
  echo "Branch '$BRANCH_NAME' does not exist. Proceeding..."
# Create and switch to the new branch
    echo "Creating a new branch '$BRANCH_NAME'..."
    git checkout -b "$BRANCH_NAME"
fi


# Stage all changes
echo "Staging changes..."
git status
git add .
git status

# Commit the changes
echo "Committing changes..."
git commit -m "$COMMIT_MESSAGE"

# Push the branch to the remote repository
echo "Pushing branch '$BRANCH_NAME' to remote..."
git push origin "$BRANCH_NAME"

# Create a Pull Request using GitHub CLI
echo "Creating a Pull Request..."
if [ -z "$PR_DESCRIPTION" ]; then
  gh pr create --title "$PR_TITLE" --body "" --base main --head "$BRANCH_NAME"
else
  gh pr create --title "$PR_TITLE" --body "$PR_DESCRIPTION" --base main --head "$BRANCH_NAME"
fi

echo "Pull Request created successfully!"
