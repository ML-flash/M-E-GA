# Contributing to Mutable Encoding enabled Genetic Algorithm (MEGA)

Thank you for your interest in contributing to MEGA! We use a **GitFlow**-inspired workflow on **GitHub**, so please read on to get an overview of how you can participate effectively.

---

## Table of Contents
1. [Overview of GitFlow](#overview-of-gitflow)
2. [Repo Owner: Initial Setup](#repo-owner-initial-setup)
3. [Forking & Local Setup (Contributors)](#forking--local-setup-contributors)
4. [Working on a Feature](#working-on-a-feature)
5. [Committing & Squashing](#committing--squashing)
6. [Creating a Pull Request (PR)](#creating-a-pull-request-pr)
7. [Repo Owner: Reviewing & Merging PRs](#repo-owner-reviewing--merging-prs)
8. [Merging to Main/Production](#merging-to-mainproduction)
9. [Code Style & Standards](#code-style--standards)
10. [Getting Help](#getting-help)

---

## Overview of GitFlow
Under GitFlow, we maintain two long-running branches in the official repository:
- **main**: The production/stable branch. We only merge into `main` when we are ready to create a release or a stable milestone.
- **dev**: The day-to-day development branch. All feature branches fork off of `dev`. When a feature or bugfix is complete, we merge its branch back into `dev`. Then eventually, when `dev` is stable, we merge `dev` into `main` for an official release.

This allows us to keep **`main`** clean and production-ready while ongoing work happens in **`dev`**.

---

## Repo Owner: Initial Setup
*(If you’re not the repo owner, you can skip this section.)*

1. **Create the `dev` branch**  
   - On GitHub, navigate to your repository’s main page.  
   - Make sure you already have `main` or `master` as your default.  
   - If `dev` doesn’t exist yet, click “Branch,” type `dev`, and create it.  

2. **(Optional) Set `dev` as the default branch**  
   - You can leave `main` as default or set `dev` as default if you prefer new contributors to land there. Go to your repo **Settings** > **Branches** > **Default branch**.

3. **Branch Protections** (optional but recommended)  
   - Consider enabling “Require pull request reviews before merging” for both `main` and `dev`. This helps keep you from accidentally merging incomplete code.  

That’s it! Now you have a `dev` branch to support GitFlow.

---

## Forking & Local Setup (Contributors)
All external contributors should **fork** the official repository and clone their fork.

1. **Fork the Repo**  
   - Go to the official MEGA repository on GitHub.  
   - Click the “Fork” button (top-right corner) to create your personal copy.  

2. **Clone Your Fork**  
   ```bash
   git clone https://github.com/YourUsername/M-E-GA.git
   cd M-E-GA
   ```
*Replace the above URL with your fork’s URL.*

3. **Set Up a Virtual Environment** (recommended for Python)
   ```bash
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   ```

4. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   # or pip install .
   ```

5. **Add the Official Repo as an Upstream** (so you can pull changes from it later)
   ```bash
   git remote add upstream https://github.com/OfficialOrg/M-E-GA.git
   ```
   *Adjust the URL to point to the official repository.*

6. **Sync with Upstream**  
   If you haven’t already, update your local fork’s branches:
   ```bash
   git checkout dev
   git pull upstream dev
   ```
   to ensure you have the latest from the official repo’s `dev`.

---

## Working on a Feature
### 1. Create a Feature Branch
- Always branch **off** of `dev`, never `main`.
- **Example**:
  ```bash
  git checkout dev
  git pull upstream dev   # Make sure dev is up to date
  git checkout -b my-new-feature
  ```

### 2. Make Your Changes
- Write code, add docstrings, write tests, etc.
- Keep commits small and logically grouped if possible.

### 3. Check & Test
- Run the test suite to confirm nothing’s broken:
  ```bash
  pytest
  ```
  or if you have a different test framework, use that.

---

## Committing & Squashing
We require **commit-squashing** to keep the history tidy:
1. **Commit frequently** while you work, but:
   ```bash
   git add .
   git commit -m "Implement partial mutation changes"
   ```
2. **When you’re finished** (or preparing to open a PR), squash those commits locally. For instance:
   ```bash
   git rebase -i HEAD~<number_of_commits_to_squash>
   ```
   Replace `pick` with `squash` or `fixup` for all but the first commit, then save.

3. **Review** your commit log with `git log`. Ideally, you’ll see just one or a few commits per feature.

---

## Creating a Pull Request (PR)
1. **Push** Your Branch to Your Fork
   ```bash
   git push origin my-new-feature
   ```
2. **Open a PR on GitHub**
   - Go to your fork on GitHub, switch to the `my-new-feature` branch.
   - Click “Contribute” > “Open pull request.”
   - **Set the base branch** to `dev` on the official repository.
   - Fill out the description, mention any related issues or tickets.

3. **Wait for Review**
   - The repo owner or other maintainers may comment.
   - If changes are required, make them locally, squash if needed, and push again.

*(Remember, we do not merge directly into `main`. We merge into `dev` with the PR.)*

---

## Repo Owner: Reviewing & Merging PRs
*(These steps are for the person who maintains the official repo.)*

1. **Open the PR** on your GitHub repository’s “Pull Requests” tab.
2. **Review Changes**
   - You can comment on specific lines, request changes, or approve.
3. **Ensure Commits are Squashed**
   - Check the contributor’s commits. If not squashed, you can ask them to do so or use GitHub’s “Squash and merge” option.
4. **Merge**
   - Once you’re happy with the feature branch, click “Merge pull request” > “Squash and merge” (or “Rebase and merge” if that’s your chosen style).
   - This merges the feature branch into `dev` (not `main`).

---

## Merging to Main/Production
Over time, multiple features might accumulate in `dev`. **When you’re ready for a release**:

1. **Create a PR from `dev` to `main`**
   - On GitHub, compare `dev` as the head branch and `main` as the base.
2. **Review** any final changes, and once stable, **merge**.
3. **Tag a Release** (optional but recommended).
   - E.g. `git tag v1.2.3` then `git push --tags`.

---

## Code Style & Standards
1. **Python style**: We generally follow [PEP 8](https://peps.python.org/pep-0008/).
2. **Docstrings**: Add them to your functions and classes describing parameters, returns, etc.
3. **Testing**: Each feature or bugfix should come with relevant tests in `tests/`.
4. **Single Responsibility**: Keep functions and classes focused. This helps with readability and testing.
5. **Changelog**: Append the `CHANGELOG.md` file with a description of the changes your branch made.

---

## Getting Help
- **Open an Issue** on the GitHub repository if you run into trouble.
- **Join our Discord** ([link](https://discord.gg/v6DfrEYDXW)) for more interactive support or discussion.

**We truly appreciate your help!** By following this flow, we ensure our `main` branch stays stable, while new goodies are developed on `dev`. Thanks again, and happy coding!
