from .base import BaseParser


class CommonParser(BaseParser):
    def __init__(self, category="articles"):
        self.category = category

    def parse(self, html: str, url: str) -> dict:
        soup = self._get_soup(html)
        title = soup.find("title")
        title_text = self._clean_text(title.text) if title else "Unknown"

        content_div = soup.find(class_="page-content") or soup.find("article") or soup.body
        content_text = self._html_to_markdown(content_div)

        return {
            "url": url,
            "title": title_text,
            "category": self.category,
            "content": f"# {title_text}\n\n{content_text}"
        }