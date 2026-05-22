import asyncio
from core.config_manager import ConfigManager
from core.token_manager import TokenManager

async def main():
    config = ConfigManager(config_path="config.yaml")
    token_mgr = TokenManager(config)
    print("Refreshing token...")
    await token_mgr.refresh_token()
    print(f"Refreshed token: {token_mgr.get_token()}")

if __name__ == "__main__":
    asyncio.run(main())
