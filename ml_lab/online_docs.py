"""Bounded, asynchronous extraction from allowlisted official documentation pages."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urljoin,urlsplit,urldefrag
import threading
import requests
import streamlit as st

HOSTS={'scikit-learn.org','docs.pytorch.org','catboost.ai','lightgbm.readthedocs.io','xgboost.readthedocs.io'}
IMAGE_HOSTS=HOSTS|{'raw.githubusercontent.com'}


class Node:
    def __init__(self,tag='',attrs=None): self.tag=tag;self.attrs=dict(attrs or []);self.children=[]
    def descendants(self):
        yield self
        for child in self.children:
            if isinstance(child,Node): yield from child.descendants()
    def text(self):
        if self.tag in ('script','style','nav','header','footer'): return ''
        return ' '.join(child.text() if isinstance(child,Node) else child for child in self.children)


class DocumentParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.root=Node();self.stack=[self.root]
    def handle_starttag(self,tag,attrs):
        node=Node(tag,attrs);self.stack[-1].children.append(node)
        if tag not in ('area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'): self.stack.append(node)
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if self.stack[-1].tag==tag: self.stack.pop()
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:
                del self.stack[i:];break
    def handle_data(self,data): self.stack[-1].children.append(data)


def official_url(url,image=False):
    try:
        parsed=urlsplit(url)
        return parsed.scheme=='https' and parsed.hostname in (IMAGE_HOSTS if image else HOSTS) and not parsed.username and not parsed.password and parsed.port in (None,443)
    except ValueError: return False


def parse_document(content,url):
    parser=DocumentParser();parser.feed(content)
    all_nodes=list(parser.root.descendants())
    anchor=urlsplit(url).fragment
    root=next((node for node in all_nodes if anchor and node.attrs.get('id')==anchor),None)
    if root is None:
        root=next((node for node in all_nodes if node.tag in ('main','article') or node.attrs.get('role')=='main'),parser.root)
    nodes=list(root.descendants())
    title=next((' '.join(node.text().split()).replace('¶','').replace('#','').strip() for node in nodes if node.tag in ('h1','h2','h3')),'Официальная документация')
    paragraphs=[]
    for node in nodes:
        if node.tag!='p': continue
        text=' '.join(node.text().split()).replace('¶','').strip()
        if len(text)>45 and text not in paragraphs: paragraphs.append(text)
    # Short attributed extract; navigation, scripts and executable markup are never rendered.
    words=' '.join(paragraphs).split()
    text=' '.join(words[:160])+(' …' if len(words)>160 else '')
    images=[]
    for node in nodes:
        if node.tag!='img': continue
        src=node.attrs.get('src') or node.attrs.get('data-src')
        if not src: continue
        image_url=urljoin(url,src)
        alt=node.attrs.get('alt','')
        if not official_url(image_url,image=True) or any(word in (src+' '+alt).lower() for word in ('logo','badge','icon','tracking','pixel')): continue
        if image_url not in [item['url'] for item in images]: images.append({'url':image_url,'alt':alt or 'Иллюстрация из документации'})
        if len(images)>=4: break
    return dict(title=title,text=text,images=images,url=url)


def fetch_document(url):
    if not official_url(url): return dict(url=url,error='Источник отсутствует в списке официальных сайтов.')
    try:
        # Validate each redirect before issuing the next request.
        current=urldefrag(url)[0]
        for _ in range(5):
            with requests.get(current,timeout=(3,6),stream=True,allow_redirects=False) as response:
                if response.is_redirect:
                    current=urljoin(current,response.headers['Location'])
                    if not official_url(current): raise ValueError('Переадресация за пределы официальных сайтов.')
                    continue
                response.raise_for_status()
                chunks=[];size=0
                for chunk in response.iter_content(65536):
                    size+=len(chunk)
                    if size>3_000_000: raise ValueError('Страница превышает лимит 3 МБ.')
                    chunks.append(chunk)
                return parse_document(b''.join(chunks).decode(response.encoding or 'utf-8',errors='replace'),current+('#'+urlsplit(url).fragment if urlsplit(url).fragment else ''))
        raise ValueError('Слишком много переадресаций.')
    except (requests.RequestException,ValueError,KeyError) as exc:
        return dict(url=url,error=str(exc))


class DocumentationLoader:
    def __init__(self):
        self.pool=ThreadPoolExecutor(max_workers=3,thread_name_prefix='official-model-docs')
        self.jobs={};self.lock=threading.Lock()
    def get(self,url,refresh=False):
        with self.lock:
            if refresh or url not in self.jobs:
                if url not in self.jobs or self.jobs[url].done(): self.jobs[url]=self.pool.submit(fetch_document,url)
            job=self.jobs[url]
        if not job.done(): return None
        try: return job.result()
        except Exception as exc: return dict(url=url,error=str(exc))


@st.cache_resource
def documentation_loader(): return DocumentationLoader()


@st.fragment(run_every='3s')
def render_online_document(url,context_key):
    document=documentation_loader().get(url)
    if document is None:
        st.caption('Загружаем текст и иллюстрации из официального источника…');return
    if document.get('error'):
        st.caption('Источник сейчас недоступен. Локальные формулы и схема модели доступны выше.')
    else:
        st.markdown('**'+document['title'].replace('*','')+'**')
        if document['text']: st.text(document['text'])
        for item in document['images']: st.image(item['url'],caption=item['alt'])
        if not document['images']: st.caption('В этом разделе документации нет извлекаемых иллюстраций. Схема устройства модели показана выше.')
        st.caption('Краткая выдержка и иллюстрации источника. Они описывают алгоритм, а не структуру конкретно обученной модели.')
    st.markdown(f'[Открыть источник]({url})')
    if st.button('Обновить материал',key='refresh_document_'+context_key):
        documentation_loader().get(url,refresh=True);st.rerun()
