# 🔑 API Key Setup Guide for Researchers

**A Step-by-Step Guide to Setting Up OpenAI API Keys for ICVision**

This guide is designed for researchers who may be new to using API keys and environment variables. Don't worry - it's easier than it looks!

## 📋 Table of Contents

- [What is an API Key?](#what-is-an-api-key)
- [Getting Your OpenAI API Key](#getting-your-openai-api-key)
- [Setting Up Your API Key](#setting-up-your-api-key)
- [Troubleshooting](#troubleshooting)
- [Security Best Practices](#security-best-practices)
- [FAQ](#faq)

## 🤔 What is an API Key?

Think of an API key like a **password that allows ICVision to talk to OpenAI's servers**.

- **API** = Application Programming Interface (how programs talk to each other)
- **Key** = A secret password that proves you have permission to use OpenAI's services

**Why do you need this?**
- ICVision sends your EEG component images to OpenAI's AI models for classification
- OpenAI needs to know who is making the request (for billing and security)
- The API key is your "digital signature" that identifies your account

## 🔐 Getting Your OpenAI API Key

### Step 1: Create an OpenAI Account

1. **Go to OpenAI's website**: https://platform.openai.com/
2. **Click "Sign up"** (or "Log in" if you already have an account)
3. **Create your account** with email and password
4. **Verify your email** when prompted

### Step 2: Add Payment Information

⚠️ **Important**: OpenAI requires a payment method even for small usage amounts.

1. **Go to "Billing"** in your OpenAI dashboard
2. **Add a payment method** (credit card)
3. **Set up billing limits** (recommended: start with $5-10/month)

**Cost Estimate**: Processing 100 EEG components typically costs $0.50-$2.00

### Step 3: Generate Your API Key

1. **Navigate to "API Keys"** in the left sidebar
2. **Click "Create new secret key"**
3. **Give it a name** (e.g., "ICVision Research Key")
4. **Copy the key immediately** - you won't be able to see it again!

**Your API key will look like this:**
```
sk-proj-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx1234yz567890ab
```

## 🔧 Setting Up Your API Key

You have three options for providing your API key to ICVision. Choose the one that works best for you:

### Option 1: Environment Variable (Recommended)

This is the **safest and most convenient** method.

#### On Windows:

**Method A: Using Command Prompt**
1. **Press `Win + R`**, type `cmd`, press Enter
2. **Type this command** (replace `your_actual_key` with your real key):
   ```cmd
   setx OPENAI_API_KEY "sk-proj-your_actual_key_here"
   ```
3. **Close and reopen** any command prompts or applications
4. **Test it worked**:
   ```cmd
   echo %OPENAI_API_KEY%
   ```

**Method B: Using System Settings (Permanent)**
1. **Press `Win + X`** → "System"
2. **Click "Advanced system settings"**
3. **Click "Environment Variables"**
4. **Under "User variables"**, click "New"
5. **Variable name**: `OPENAI_API_KEY`
6. **Variable value**: `sk-proj-your_actual_key_here`
7. **Click OK** to save
8. **Restart your computer** or application

#### On Mac/Linux:

**Method A: Temporary (for current session only)**
```bash
export OPENAI_API_KEY="sk-proj-your_actual_key_here"
```

**Method B: Permanent (recommended)**
1. **Open Terminal**
2. **Edit your shell profile**:
   ```bash
   # For newer Macs (zsh):
   echo 'export OPENAI_API_KEY="sk-proj-your_actual_key_here"' >> ~/.zshrc

   # For older Macs/Linux (bash):
   echo 'export OPENAI_API_KEY="sk-proj-your_actual_key_here"' >> ~/.bashrc
   ```
3. **Reload your shell**:
   ```bash
   source ~/.zshrc  # or ~/.bashrc
   ```
4. **Test it worked**:
   ```bash
   echo $OPENAI_API_KEY
   ```

### Option 2: .env File (Good for Projects)

This method keeps your API key in a special file that ICVision can read.

1. **Navigate to your project folder** (where you'll run ICVision)
2. **Create a new file** called `.env` (note the dot at the beginning)
3. **Add this line** to the file:
   ```
   OPENAI_API_KEY=sk-proj-your_actual_key_here
   ```
4. **Save the file**

**⚠️ Important**: Never share or commit the `.env` file to version control!

### Option 3: Command Line (Quick Testing)

You can provide the API key directly when running ICVision:

```bash
icvision data.set ica.fif --api-key sk-proj-your_actual_key_here
```

**Note**: This method is less secure because the key appears in your command history.

## 🧪 Testing Your Setup

### Test 1: Check Environment Variable

**Windows:**
```cmd
echo %OPENAI_API_KEY%
```

**Mac/Linux:**
```bash
echo $OPENAI_API_KEY
```

**Expected Result**: You should see your API key printed out.

### Test 2: Test with ICVision

```bash
# Try running ICVision (it will tell you if the API key is working)
icvision --help
```

### Test 3: Quick API Test

Create a simple test script to verify your API key works:

```python
# test_api.py
import os

# Check if API key is available
api_key = os.environ.get('OPENAI_API_KEY')

if api_key:
    print("✅ API key found!")
    print(f"Key starts with: {api_key[:15]}...")

    # Test if we can import OpenAI (optional)
    try:
        import openai
        openai.api_key = api_key
        print("✅ OpenAI library is ready!")
    except ImportError:
        print("⚠️  OpenAI library not installed (this is OK if using ICVision)")
else:
    print("❌ No API key found!")
    print("Please set the OPENAI_API_KEY environment variable.")
```

Run it with:
```bash
python test_api.py
```

## 🚨 Troubleshooting

### Problem: "No API key found" error

**Symptoms**: ICVision says it can't find your API key

**Solutions**:
1. **Check spelling**: Make sure it's exactly `OPENAI_API_KEY` (all caps)
2. **Restart applications**: Close and reopen your terminal/command prompt
3. **Check the key**: Verify your API key is correctly set:
   - Windows: `echo %OPENAI_API_KEY%`
   - Mac/Linux: `echo $OPENAI_API_KEY`
4. **Try the direct method**: Use `--api-key` flag as a test

### Problem: "Invalid API key" error

**Symptoms**: ICVision rejects your API key

**Solutions**:
1. **Check for extra spaces**: Make sure there are no spaces before/after your key
2. **Regenerate key**: Go to OpenAI dashboard and create a new key
3. **Check account status**: Ensure your OpenAI account is active and has billing set up

### Problem: "Rate limit exceeded" error

**Symptoms**: OpenAI says you're making too many requests

**Solutions**:
1. **Wait a few minutes** and try again
2. **Reduce batch size**: Use `--batch-size 5` instead of default 10
3. **Reduce concurrency**: Use `--max-concurrency 2` instead of default 4

### Problem: "Insufficient credits" error

**Symptoms**: OpenAI says you don't have enough credits

**Solutions**:
1. **Check your billing**: Go to OpenAI dashboard → Billing
2. **Add credits**: Purchase additional credits or increase your limit
3. **Check usage**: Review your current usage and limits

### Problem: Environment variable not working

**Symptoms**: You set the variable but ICVision still can't find it

**Solutions**:

**Windows**:
1. **Restart your computer** (sometimes required for system-wide variables)
2. **Use PowerShell instead**:
   ```powershell
   $env:OPENAI_API_KEY="sk-proj-your_key_here"
   ```
3. **Check user vs system variables**: Make sure you set it in the right place

**Mac/Linux**:
1. **Check your shell**: Run `echo $SHELL` to see if you're using bash or zsh
2. **Edit the right file**:
   - Zsh (newer Macs): `~/.zshrc`
   - Bash (older Macs/Linux): `~/.bashrc`
3. **Reload the file**: `source ~/.zshrc` or `source ~/.bashrc`

## 🔒 Security Best Practices

### ✅ DO:
- **Keep your API key secret** - never share it with others
- **Use environment variables** instead of putting keys in code
- **Set spending limits** in your OpenAI account
- **Monitor your usage** regularly
- **Regenerate keys** if you think they've been compromised

### ❌ DON'T:
- **Never put API keys in code** that you'll share or commit to Git
- **Don't share screenshots** that show your full API key
- **Don't email or message** your API key to others
- **Don't use the same key** for multiple unrelated projects

### 🛡️ Extra Security Tips:
1. **Create separate keys** for different projects
2. **Name your keys descriptively** (e.g., "Lab Computer - EEG Analysis")
3. **Delete unused keys** from your OpenAI dashboard
4. **Use .gitignore** to exclude `.env` files from version control:
   ```gitignore
   # Add to .gitignore file
   .env
   *.env
   ```

## ❓ FAQ

### Q: How much does it cost to use ICVision?
**A**: Typically $0.50-$2.00 per 100 EEG components. The exact cost depends on:
- Number of components to classify
- Model used (gpt-4.1 is more expensive but more accurate)
- Image size and complexity

### Q: Can I use a free OpenAI account?
**A**: No, OpenAI requires a paid account for API access. However, you only pay for what you use, and research usage is typically very affordable.

### Q: What if I'm at a university - can I use institutional billing?
**A**: Check with your IT department or research administrator. Some universities have:
- Institutional OpenAI accounts
- Research computing credits that cover API costs
- Special pricing for academic use

### Q: Can multiple people share one API key?
**A**: Technically yes, but it's not recommended because:
- You can't track individual usage
- One person's heavy usage affects everyone's rate limits
- Security risk if the key needs to be changed

### Q: How do I know if my API key is working?
**A**: Run this simple test:
```bash
icvision --help
```
If you see the help text without API key errors, you're all set!

### Q: What if I accidentally shared my API key?
**A**:
1. **Immediately go to OpenAI dashboard** and delete the compromised key
2. **Create a new key**
3. **Update your environment variable** with the new key
4. **Monitor your usage** for any unexpected activity

### Q: Can I use ICVision offline?
**A**: No, ICVision requires an internet connection to communicate with OpenAI's servers for component classification.

### Q: What models should I use?
**A**: For research:
- **gpt-4.1**: Best accuracy, higher cost (recommended for final analyses)
- **gpt-4.1-mini**: Good accuracy, lower cost (good for testing)

## 📞 Getting Help

If you're still having trouble:

1. **Check the ICVision documentation**: https://cincibrainlab.github.io/ICVision/
2. **Look at ICVision issues**: https://github.com/cincibrainlab/ICVision/issues
3. **Contact your IT support** for environment variable help
4. **Ask your research supervisor** if they have experience with API keys

**For OpenAI-specific issues**:
- OpenAI Help Center: https://help.openai.com/
- OpenAI Status Page: https://status.openai.com/

---

**Remember**: Setting up API keys might seem intimidating at first, but once it's done, ICVision will work seamlessly! Take your time, follow the steps carefully, and don't hesitate to ask for help if needed. 🚀
