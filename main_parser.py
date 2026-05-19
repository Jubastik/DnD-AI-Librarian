import asyncio
from src.parser.crawler import Crawler

if __name__ == "__main__":
    crawler = Crawler()
    try:
        # Для Windows может понадобиться policy
        # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(crawler.run(retry=3))
    except KeyboardInterrupt:
        print("Stopped by user")