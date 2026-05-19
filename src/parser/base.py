from bs4 import BeautifulSoup, Tag, NavigableString
import re


class BaseParser:
    def parse(self, html: str, url: str) -> dict:
        """Абстрактный метод, который должен вернуть словарь с данными"""
        raise NotImplementedError

    def _get_soup(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "lxml")
        for trash in soup(["script", "style", "nav", "footer", "iframe", "div.adb-in-content", "noindex"]):
            trash.decompose()
        return soup

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace('\xa0', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    def _html_to_markdown(self, element) -> str:
        """Рекурсивно превращает HTML элемент в Markdown-подобный текст"""
        if element is None:
            return ""

        text_parts = []

        if isinstance(element, NavigableString):
            text = str(element).strip()
            return text if text else ""

        if isinstance(element, Tag):
            # Заголовки
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                header_text = element.get_text(strip=True)
                return f"\n\n{'#' * int(element.name[1])} {header_text}\n"

            if element.name in ['p', 'div', 'li']:
                content = "".join([self._html_to_markdown(child) for child in element.children])
                prefix = "- " if element.name == 'li' else ""
                return f"\n{prefix}{content.strip()}\n"

            if element.name == 'table':
                return self._parse_table(element)

            if element.name == 'a':
                return element.get_text(strip=True)

            for child in element.children:
                text_parts.append(self._html_to_markdown(child))

        return " ".join(text_parts)

    def _parse_table(self, table_tag: Tag) -> str:
        """Превращает HTML таблицу в Markdown таблицу"""
        rows = []

        headers = [th.get_text(strip=True) for th in table_tag.find_all('th')]
        if headers:
            rows.append("| " + " | ".join(headers) + " |")
            rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for tr in table_tag.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            if cells:
                if len(cells) == len(headers) or not headers:
                    rows.append("| " + " | ".join(cells) + " |")

        return "\n" + "\n".join(rows) + "\n"
