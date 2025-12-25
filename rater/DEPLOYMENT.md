# ICVision Rater Deployment Guide

Deployment using Kamal 2.8+ with PostgreSQL accessory.

## Prerequisites

- Kamal 2.8+ installed (`gem install kamal`)
- Docker on deployment server
- SSH access to server (root or user with Docker permissions)

## Quick Start

```bash
# 1. Update server IP in config/deploy.yml
#    Replace 192.168.0.1 with your server IP

# 2. Set PostgreSQL password (use a strong password in production)
export POSTGRES_PASSWORD=$(openssl rand -base64 32)

# 3. First-time setup
kamal setup

# 4. Subsequent deploys
kamal deploy
```

## Configuration

### Server Setup

Edit `config/deploy.yml`:

```yaml
servers:
  web:
    - YOUR_SERVER_IP

accessories:
  postgres:
    host: YOUR_SERVER_IP
```

### SSL/HTTPS

For production with a real domain:

```yaml
proxy:
  ssl: true
  host: rater.yourdomain.com
```

Ensure `config.force_ssl = true` in `config/environments/production.rb`.

### Secrets

Edit `.kamal/secrets` to configure:
- `RAILS_MASTER_KEY` - from `config/master.key`
- `POSTGRES_PASSWORD` - database password

For 1Password integration:
```bash
SECRETS=$(kamal secrets fetch --adapter 1password --account your-account --from Vault/Item RAILS_MASTER_KEY POSTGRES_PASSWORD)
```

## Commands

```bash
# Deploy
kamal deploy

# View logs
kamal logs

# Rails console
kamal console

# Database console
kamal dbc

# Shell access
kamal shell

# Restart app
kamal app restart

# Reboot proxy (after Kamal upgrade)
kamal proxy reboot
```

## Database Management

```bash
# Run migrations
kamal app exec "bin/rails db:migrate"

# Database backup
kamal accessory exec postgres "pg_dump -U rater rater_production" > backup.sql

# Restore from backup
cat backup.sql | kamal accessory exec -i postgres "psql -U rater rater_production"
```

## Troubleshooting

### Proxy Issues
```bash
# Check proxy status
kamal proxy status

# Reboot proxy (requires kamal-proxy v0.9.0+)
kamal proxy reboot
```

### Container Issues
```bash
# View container details
kamal details

# Check app health
kamal app exec "curl -s localhost:80/up"
```

## Kamal 2.8+ Notes

- Uses kamal-proxy (not Traefik)
- Default app port is 80 (Thruster handles this)
- Containers run in Docker network `kamal`
- Local registry support for faster deploys
- Secrets managed via `.kamal/secrets` (not `.env`)

## Version Requirements

- Kamal: 2.8.0+
- kamal-proxy: 0.9.0+
- Ruby: 3.4.7
- Rails: 8.1.1
- PostgreSQL: 17
