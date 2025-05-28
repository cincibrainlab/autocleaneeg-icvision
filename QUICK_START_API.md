# 🚀 Quick Start: API Setup for Researchers

**New to API keys? This 5-minute guide gets you started with ICVision!**

## 🎯 What You Need

1. **OpenAI Account** (free to create)
2. **Credit Card** (for OpenAI billing - you only pay for what you use)
3. **5 minutes** to set everything up

**Typical Cost**: $0.50-$2.00 for 100 EEG components

## 📝 Step-by-Step Setup

### Step 1: Get Your API Key (2 minutes)

1. **Go to**: https://platform.openai.com/
2. **Sign up** or log in
3. **Add billing info**: Go to "Billing" → Add payment method
4. **Create API key**: Go to "API Keys" → "Create new secret key"
5. **Copy the key** immediately (starts with `sk-proj-...`)

### Step 2: Set Your API Key (2 minutes)

Choose your operating system:

#### Windows:
**Option A (Easiest)**: Command Prompt
```cmd
setx OPENAI_API_KEY "sk-proj-your_actual_key_here"
```
Then restart your command prompt.

**Option B**: System Settings
1. Press `Win + X` → "System"
2. "Advanced system settings" → "Environment Variables"
3. "New" → Name: `OPENAI_API_KEY`, Value: `sk-proj-your_key_here`

#### Mac/Linux:
```bash
# Add to your shell profile (permanent)
echo 'export OPENAI_API_KEY="sk-proj-your_actual_key_here"' >> ~/.zshrc

# Reload your shell
source ~/.zshrc
```

### Step 3: Test It Works (1 minute)

```bash
# Check your API key is set
echo $OPENAI_API_KEY  # Mac/Linux
echo %OPENAI_API_KEY%  # Windows

# Test ICVision
icvision --help
```

**Expected Result**: You should see your API key and ICVision help text.

## 🧪 Run Your First Analysis

```bash
# Basic command (replace with your file paths)
icvision /path/to/raw_data.set /path/to/ica_data.fif

# With custom settings
icvision /path/to/raw_data.set /path/to/ica_data.fif \
    --output-dir my_results/ \
    --confidence-threshold 0.8 \
    --verbose
```

## 🚨 Quick Troubleshooting

**"No API key found"**:
- Check spelling: `OPENAI_API_KEY` (exactly)
- Restart your terminal/command prompt
- Try: `icvision data.set ica.fif --api-key sk-proj-your_key_here`

**"Invalid API key"**:
- Regenerate key at https://platform.openai.com/api-keys
- Check for extra spaces in your key
- Verify billing is set up

**"Rate limit exceeded"**:
- Wait a few minutes
- Use: `--batch-size 5 --max-concurrency 2`

## 💰 Managing Costs

**Set Usage Limits** (Recommended):
1. Go to OpenAI dashboard → "Billing" → "Usage limits"
2. Set monthly limit (e.g., $10)
3. Enable email alerts

**Cost-Saving Tips**:
- Start with small datasets to test
- Use `--model gpt-4-vision-preview` for testing (cheaper)
- Monitor usage in OpenAI dashboard

## 🔗 Full Documentation

- **Complete Setup Guide**: [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md)
- **Installation Guide**: [docs/installation.rst](docs/installation.rst)
- **Troubleshooting**: [docs/troubleshooting.rst](docs/troubleshooting.rst)
- **Online Docs**: https://cincibrainlab.github.io/ICVision/

## 📞 Need Help?

- **GitHub Issues**: https://github.com/cincibrainlab/ICVision/issues
- **OpenAI Help**: https://help.openai.com/

---

**You're Ready!** Once set up, ICVision will automatically use your API key for all analyses. No need to re-enter it! 🎉
