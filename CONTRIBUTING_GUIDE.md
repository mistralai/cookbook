# Contributing Guide

Thank you for your interest in contributing to this project! Please follow the guidelines below to ensure a smooth contribution process.

## 🔒 Security Requirements

This repository has security checks in place to prevent accidental commits of private keys and sensitive information.

### Required Setup

Before making any commits, you **must** install pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks for this repository
pre-commit install
```

### What happens if you don't?

- Your commits will be **blocked locally** if they contain private keys
- PRs with private keys will **fail CI checks** and cannot be merged
- This protects everyone from accidentally exposing sensitive information

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/<YOUR-USERNAME>/cookbook.git
   cd cookbook
   ```

3. **Set up pre-commit hooks** (required!)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Create a new branch**
   ```bash
   git checkout -b <YOUR BRANCH NAME>
   ```

5. **Make your changes**

6. **Test your changes**
   ```bash
   # Run security check manually (optional)
   pre-commit run detect-private-key --all-files
   ```

7. **Commit and push**
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin <YOUR BRANCH NAME>
   ```

8. **Create a Pull Request**

## ⚠️ Common Issues

### "Private key detected" error
If you see this error when committing:
```
detect-private-key...........................................................Failed
```

- Remove any private keys, API keys, or sensitive information from your files
- Check files like `.env`, config files, or any files containing credentials
- Use environment variables or secure vaults for sensitive data instead

## Best Practices

- Never commit private keys, API keys, passwords, or other sensitive information
- Use environment variables for configuration
- Keep sensitive data in `.env` files and add them to `.gitignore`
- Test your changes locally before creating a PR