#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if the current directory is a git repository
git branch


echo "Enter exists or new branch name:"
read BRANCH_NAME

echo "Enter commit message:"
read COMMIT_MESSAGE

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

echo "Pull Request created successfully!"
