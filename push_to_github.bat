@echo off
REM Script to push code to GitHub

echo Initializing Git repository...
git init

echo Adding all files...
git add .

echo Creating initial commit...
git commit -m "Initial commit: AgriVision AI project with data structures"

echo Setting main branch...
git branch -M main

echo Please enter your GitHub username:
set /p USERNAME=

echo Adding remote repository...
git remote add origin https://github.com/%USERNAME%/agrivision-ai.git

echo Pushing to GitHub (this will replace existing content)...
git push -u origin main --force

echo.
echo Done! Your code has been pushed to GitHub.
echo Repository URL: https://github.com/%USERNAME%/agrivision-ai
pause
