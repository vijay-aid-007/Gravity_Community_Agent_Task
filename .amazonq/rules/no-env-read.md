# Security Rules

- NEVER read `.env` or any file containing secrets directly
- To check environment variables, only read `config/settings.py` or `.env.example`
- NEVER print or log the contents of `.env` in any response
