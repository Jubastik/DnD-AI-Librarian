from .base import BaseParser


class SpellParser(BaseParser):
    def parse(self, html: str, url: str) -> dict:
        soup = self._get_soup(html)

        title = soup.find("title")
        title_text = self._clean_text(title.text) if title else "Unknown Spell"

        description_container = soup.find("div", itemprop="description")
        if not description_container:
            description_container = soup.find(class_="card-body")

        desc_text = self._html_to_markdown(description_container)

        params_block = soup.find("ul", class_="params card__article-body")
        if params_block is None or params_block.find("li", class_="subsection desc") is None:
            # print(url, params_block)
            raise ValueError("Params block not found")
        params_block.find("li", class_="subsection desc").decompose()
        params_text = ""
        if params_block:
            params_text = self._html_to_markdown(params_block)

        full_content = f"# {title_text}\n\n## Характеристики\n{params_text}\n\n## Описание\n{desc_text}"

        return {
            "url": url,
            "title": title_text,
            "category": "spells",
            "content": full_content.strip(),
            "is_homebrew": "homebrew" in url
        }


if __name__ == "__main__":
    import requests

    url = "https://dnd.su/spells/38-greater_invisibility/"

    html = requests.get(url)

    s = SpellParser()
    res = s.parse(html.text, url)

    print(res)
