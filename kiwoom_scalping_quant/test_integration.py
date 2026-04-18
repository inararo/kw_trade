import asyncio
import logging
from core.container import Container

async def run_integration_test():
    # Setup logger
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Init Container
    container = Container()

    # 1. Test Universe Auto Generation
    print("\n--- 1. 유니버스 자동 생성 테스트 시작 ---")
    asset_vm = container.asset_data_view_model()

    # Signal listener for completion
    universe_done = asyncio.Event()
    def on_universe_done(msg):
        print(f"UI Signal (Universe Completed): {msg}")
        universe_done.set()

    asset_vm.fetch_completed.connect(on_universe_done)

    # Trigger Universe Generation
    asset_vm.build_universe()

    # Wait for completion
    await universe_done.wait()

    # 2. Test Historical Data Fetch & Throttling & InfluxDB Insert
    print("\n--- 2. 과거 데이터 수집 및 DB 적재 테스트 시작 ---")

    # Mock symbols and fetching logic inside ViewModel uses asyncio tasks.
    # We will trigger it for '005930'
    fetch_done = asyncio.Event()
    def on_fetch_done(msg):
        print(f"UI Signal (Fetch Completed): {msg}")
        fetch_done.set()

    asset_vm.fetch_completed.disconnect(on_universe_done)
    asset_vm.fetch_completed.connect(on_fetch_done)

    def on_progress(pct):
        if pct % 20 == 0:
            print(f"UI Progress Update: {pct}%")

    asset_vm.sig_progress_updated.connect(on_progress)

    # Ensure Influx client runs its background flush loop
    influx = container.influx_client()
    influx_task = asyncio.create_task(influx.start())

    # Trigger Fetch
    asset_vm.start_historical_fetch("005930", "20230101")

    # Wait for completion
    await fetch_done.wait()

    # Wait a bit for the async bulk inserts to finish logging
    await asyncio.sleep(1)

    # Stop
    await influx.close()
    influx_task.cancel()

if __name__ == "__main__":
    asyncio.run(run_integration_test())
