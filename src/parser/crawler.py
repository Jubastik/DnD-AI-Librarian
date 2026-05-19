import aiohttp
import asyncio
from bs4 import BeautifulSoup
from loguru import logger
import json
import os
import random
import time
from urllib.parse import urljoin

from src.core.config import settings
from src.parser.factory import ParserFactory

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

FILE_TTL_DAYS = 7
FILE_TTL_SECONDS = FILE_TTL_DAYS * 24 * 60 * 60


class Crawler:
    def __init__(self):
        self.headers = HEADERS
        self.semaphore = asyncio.Semaphore(10)

    async def fetch(self, session, url):
        try:
            async with session.get(url, headers=self.headers, timeout=30) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.error(f"Error {response.status} fetching {url}")
                    return None
        except Exception as e:
            logger.error(f"Exception fetching {url}: {e}")
            return None

    async def get_sitemap_links(self, session):
        """Парсит sitemap.xml и возвращает словарь {url: category}"""
        logger.info(f"Fetching sitemap: {settings.SITEMAP_URL}")
        xml_content = await self.fetch(session, settings.SITEMAP_URL)
        if not xml_content:
            return {}

        soup = BeautifulSoup(xml_content, "lxml-xml")
        links = {}

        urls = soup.find_all("loc")
        logger.info(f"Found {len(urls)} URLs in sitemap. Filtering...")

        for url_tag in urls:
            url = url_tag.text
            for category, filter_str in settings.URL_FILTERS.items():
                if filter_str in url and url.split("/")[-2] != category and "homebrew" not in url:
                    links[url] = category
                    break

        logger.info(f"Filtered down to {len(links)} target URLs")
        return links

    async def get_extra_links(self, session):
        """Сканирует страницы-хабы из конфига и достает оттуда ссылки"""
        extra_links = {}

        if not hasattr(settings, "EXTRA_HUBS") or not settings.EXTRA_HUBS:
            return extra_links

        logger.info(f"Processing {len(settings.EXTRA_HUBS)} extra hub pages...")

        for hub in settings.EXTRA_HUBS:
            url = urljoin(settings.BASE_URL, hub["url"])
            category = hub["category"]
            selector = hub["selector"]

            logger.info(f"Scanning hub: {url}")
            html = await self.fetch(session, url)

            if not html:
                continue

            soup = BeautifulSoup(html, "lxml")

            found_tags = soup.select(selector)
            count = 0

            for tag in found_tags:
                href = tag.get("href")
                if not href:
                    continue

                full_url = urljoin(url, href)

                if "#" in full_url:
                    full_url = full_url.split("#")[0]

                if "dnd.su" in full_url:
                    extra_links[full_url] = category
                    count += 1

            logger.info(f"Found {count} links in {url}")

            await asyncio.sleep(1)

        return extra_links

    async def process_url(self, session, url, category, retry=1):
        """Скачивает страницу, парсит и сохраняет в JSON"""
        # logger.info(f"Start parsing: {url}")
        async with self.semaphore:
            filename = url.replace("https://", "").replace("http://", "").replace("/", "_").strip("_") + ".json"
            filepath = os.path.join(settings.DATA_DIR, filename)

            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                last_modified = os.path.getmtime(filepath)
                age = time.time() - last_modified

                if age < FILE_TTL_SECONDS:
                    # logger.debug(f"Skipping fresh file: {filename}")
                    return
                else:
                    logger.info(f"File expired ({int(age / 86400)} days old), refreshing: {filename}")
                    need_download = True

            await asyncio.sleep(random.uniform(0.5, 1.5))

            parser = ParserFactory.get_parser(url)
            data = None

            for i in range(retry):
                html = await self.fetch(session, url)

                if not html:
                    wait_time = (i + 1) * 2
                    logger.warning(f"Attempt {i + 1} failed to fetch HTML for {url}. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                try:
                    temp_data = parser.parse(html, url)
                    if temp_data and temp_data.get("content") and temp_data.get("title"):
                        data = temp_data
                        break
                    else:
                        raise ValueError("Parsed data is empty (missing params_block?)")

                except Exception as e:
                    data = None
                    logger.warning(f"Attempt {i + 1} parsing failed for {url}: {e}")
                    if i == retry - 1:
                        with open("debug_last_fail.html", "w", encoding="utf-8") as f:
                            f.write(html)
                    await asyncio.sleep((i + 1) * 2)
            if data is None:
                logger.warning(f"Failed to download: {url}, probably a private")
                return
            if not data["content"] or not data["title"]:
                logger.warning(f"Skipping empty content: {url}")
                return

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved: {data['title']} ({category})")

    async def run(self, retry=3):
        async with aiohttp.ClientSession() as session:
            logger.info("Gathering links...")

            results = await asyncio.gather(
                self.get_sitemap_links(session),
                self.get_extra_links(session)
            )

            sitemap_links = results[0]
            hub_links = results[1]
            logger.info(f"Found {len(sitemap_links)} links in sitemap")
            logger.info(f"Found {len(hub_links)} links in extra hubs")

            all_links_map = {**sitemap_links, **hub_links}

            if not all_links_map:
                logger.error("No links found anywhere.")
                return

            all_items = list(all_links_map.items())

            BATCH_SIZE = 100
            SLEEP_TIME = 60

            logger.info(f"Total links: {len(all_items)}. Processing in batches of {BATCH_SIZE}...")

            for i in range(0, len(all_items), BATCH_SIZE):
                batch = all_items[i: i + BATCH_SIZE]

                tasks = []
                for url, category in batch:
                    tasks.append(self.process_url(session, url, category, retry))

                logger.info(f"--- Starting batch {i} - {i + len(batch)} ---")

                await asyncio.gather(*tasks)

                if i + BATCH_SIZE < len(all_items):
                    logger.info(f"Batch complete. Sleeping {SLEEP_TIME} seconds to cool down...")
                    await asyncio.sleep(SLEEP_TIME)

            logger.info("All parsing complete!")
