from .base import BaseParser


class ClassParser(BaseParser):
    def parse(self, html: str, url: str) -> dict:
        soup = self._get_soup(html)

        title = soup.find("title")
        title_text = self._clean_text(title.text) if title else "Unknown Class"

        content_div = soup.find(class_="card-wrapper") or soup.find(class_="card-body")


        content_text = self._html_to_markdown(content_div)

        return {
            "url": url,
            "title": title_text,
            "category": "classes",
            "content": f"# {title_text}\n\n{content_text}"
        }


if __name__ == "__main__":
    import requests

    url = "https://dnd.su/class/91-fighter/"

    html = requests.get(url)

    s = ClassParser()
    res = s.parse(html.text, url)

    print(res)