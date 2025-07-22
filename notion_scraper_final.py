import os
import json
import time
import re
import hashlib
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import urllib.parse

# Configuration - AGGRESSIVE DEEP SCRAPING
START_URL = "https://pumped-aura-0a4.notion.site/TruPorch-Homes-PillowPM-Company-Wiki-0d4c3c4d7c974655821a37441be43ea6"
OUTPUT_DIR = "scraper_deep"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# LLM-optimized settings - LARGE CHUNKS FOR BETTER CONTEXT
MIN_CONTENT_LENGTH = 300    # Larger minimum for meaningful content
MAX_CONTENT_LENGTH = 4000   # Much larger chunks for rich context
MAX_PAGES = 999999          # No limit - scrape everything
MAX_DEPTH = 20              # Very deep crawling
WAIT_TIME = 5               # Longer wait for dynamic content
SCROLL_PAUSE = 3            # More time for content to load

# Retry configuration
MAX_RETRIES = 3             # Number of retry attempts per URL
RETRY_DELAY_BASE = 2        # Base delay for exponential backoff (seconds)
RETRY_DELAY_MAX = 30        # Maximum retry delay
DRIVER_RESTART_THRESHOLD = 5 # Restart driver after this many consecutive failures

class DeepNotionScraper:
    def __init__(self):
        self.setup_logging()
        self.visited_urls = set()
        self.failed_urls = {}  # Now stores retry count: {url: retry_count}
        self.permanently_failed_urls = set()
        self.all_embeddings = []
        self.model = None
        self.driver = None
        self.found_links = set()
        self.consecutive_failures = 0
        
    def setup_logging(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(OUTPUT_DIR, "deep_scraper.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_driver(self):
        try:
            if self.driver:
                self.driver.quit()
                
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.set_page_load_timeout(45)
            self.driver.implicitly_wait(10)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create driver: {e}")
            return False

    def aggressive_page_load(self, url, retry_attempt=0):
        """Aggressively load page content with multiple techniques and retry logic"""
        try:
            if not self.driver and not self.create_driver():
                return None
            
            retry_info = f" (attempt {retry_attempt + 1}/{MAX_RETRIES + 1})" if retry_attempt > 0 else ""
            self.logger.info(f"Loading: {url}{retry_info}")
            
            # Try different loading strategies based on retry attempt
            if retry_attempt == 0:
                # Standard approach
                self.driver.get(url)
                time.sleep(WAIT_TIME)
            elif retry_attempt == 1:
                # More aggressive approach - clear cache and reload
                self.logger.info("Retry with cache clearing...")
                self.driver.delete_all_cookies()
                self.driver.execute_script("window.localStorage.clear();")
                self.driver.execute_script("window.sessionStorage.clear();")
                self.driver.get(url)
                time.sleep(WAIT_TIME * 1.5)
            elif retry_attempt == 2:
                # Restart driver and try again
                self.logger.info("Retry with fresh driver...")
                self.create_driver()  # This will quit old driver and create new one
                self.driver.get(url)
                time.sleep(WAIT_TIME * 2)
            else:
                # Final attempt with maximum delays
                self.logger.info("Final retry attempt with maximum delays...")
                self.driver.get(url)
                time.sleep(WAIT_TIME * 3)
            
            # Wait for basic page structure with different strategies per retry
            wait_time = 15 + (retry_attempt * 5)  # Increase wait time with retries
            
            if 'notion.site' in url:
                # Wait for main content with increasing timeout
                try:
                    WebDriverWait(self.driver, wait_time).until(
                        EC.any_of(
                            EC.presence_of_element_located((By.TAG_NAME, "main")),
                            EC.presence_of_element_located((By.CLASS_NAME, "notion-page-content")),
                            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-block-id]")),
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".notion-selectable")),
                            EC.presence_of_element_located((By.TAG_NAME, "body"))  # Fallback
                        )
                    )
                except Exception as e:
                    self.logger.warning(f"Wait failed for {url} on attempt {retry_attempt + 1}: {e}")
                    if retry_attempt < MAX_RETRIES:
                        return None  # Signal for retry
                
                # Aggressive scrolling and interaction (more aggressive on retries)
                scroll_attempts = 5 + retry_attempt * 2
                for i in range(scroll_attempts):
                    # Scroll to different positions
                    self.driver.execute_script(f"window.scrollTo(0, {i * 500});")
                    time.sleep(1 + retry_attempt * 0.5)
                    
                    # Try clicking expandable elements
                    try:
                        expandable = self.driver.find_elements(By.CSS_SELECTOR, 
                            ".notion-toggle, .notion-callout, [aria-expanded='false'], .notion-collection-view-tab, button")
                        for elem in expandable[:10 + retry_attempt * 5]:
                            try:
                                if elem.is_displayed() and elem.is_enabled():
                                    self.driver.execute_script("arguments[0].click();", elem)
                                    time.sleep(0.5)
                            except:
                                pass
                    except:
                        pass
                
                # Final full scroll with extra time on retries
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(SCROLL_PAUSE + retry_attempt)
                
                # Try to load any lazy content (more attempts on retries)
                try:
                    lazy_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-src], [loading='lazy'], .notion-lazy")
                    for elem in lazy_elements[:20 + retry_attempt * 10]:
                        try:
                            self.driver.execute_script("arguments[0].scrollIntoView();", elem)
                            time.sleep(0.2 + retry_attempt * 0.1)
                        except:
                            pass
                except:
                    pass
                
                # Back to top
                self.driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(2)
            
            # Validate that we got actual content
            page_source = self.driver.page_source
            if len(page_source) < 500:  # Very small page might indicate loading failure
                self.logger.warning(f"Suspiciously small page source ({len(page_source)} chars) for {url}")
                if retry_attempt < MAX_RETRIES:
                    return None  # Signal for retry
            
            self.consecutive_failures = 0  # Reset on success
            return page_source
            
        except Exception as e:
            self.logger.error(f"Failed to load {url} on attempt {retry_attempt + 1}: {e}")
            return None

    def should_retry_url(self, url):
        """Check if URL should be retried"""
        if url in self.permanently_failed_urls:
            return False
        
        retry_count = self.failed_urls.get(url, 0)
        return retry_count < MAX_RETRIES

    def calculate_retry_delay(self, attempt):
        """Calculate exponential backoff delay"""
        delay = min(RETRY_DELAY_BASE ** attempt, RETRY_DELAY_MAX)
        return delay

    def extract_all_links(self, soup, base_url):
        """Extract ALL possible links with multiple strategies"""
        links = set()
        base_domain = urllib.parse.urlparse(START_URL).netloc
        
        # Strategy 1: Standard href links
        for a in soup.find_all('a', href=True):
            href = a.get('href', '').strip()
            if href and not href.startswith('#'):
                full_url = urllib.parse.urljoin(base_url, href)
                if self.is_valid_link(full_url, base_domain):
                    links.add(full_url)
        
        # Strategy 2: Notion-specific data attributes
        for elem in soup.find_all(attrs={"data-href": True}):
            href = elem.get('data-href', '').strip()
            if href:
                full_url = urllib.parse.urljoin(base_url, href)
                if self.is_valid_link(full_url, base_domain):
                    links.add(full_url)
        
        # Strategy 3: Look for Notion page IDs in various attributes
        notion_patterns = [
            r'[a-f0-9]{32}',  # 32-char hex IDs
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',  # UUID format
        ]
        
        for pattern in notion_patterns:
            # Check all text content for page IDs
            page_text = soup.get_text()
            for match in re.finditer(pattern, page_text):
                page_id = match.group()
                potential_url = f"https://{base_domain}/{page_id}"
                if self.is_valid_link(potential_url, base_domain):
                    links.add(potential_url)
        
        # Strategy 4: Look in script tags for URLs
        for script in soup.find_all('script'):
            script_text = script.get_text()
            if script_text:
                # Look for URLs in JavaScript
                url_matches = re.findall(r'https?://[^\s"\']+', script_text)
                for url_match in url_matches:
                    if base_domain in url_match and self.is_valid_link(url_match, base_domain):
                        links.add(url_match)
        
        # Strategy 5: Extract from CSS and style attributes
        for elem in soup.find_all(attrs={"style": True}):
            style = elem.get('style', '')
            url_matches = re.findall(r'url\(["\']?(https?://[^)"\']+)["\']?\)', style)
            for url_match in url_matches:
                if base_domain in url_match and self.is_valid_link(url_match, base_domain):
                    links.add(url_match)
        
        new_links = [link for link in links if (
            link not in self.visited_urls and 
            link not in self.permanently_failed_urls and
            self.should_retry_url(link)
        )]
        
        self.logger.info(f"Found {len(new_links)} new/retryable links from {base_url}")
        
        return list(new_links)

    def is_valid_link(self, url, base_domain):
        """Check if link is valid for scraping"""
        try:
            parsed = urllib.parse.urlparse(url)
            
            # Must be from same domain
            if parsed.netloc != base_domain:
                return False
            
            # Skip certain file types
            if parsed.path.lower().endswith(('.pdf', '.jpg', '.png', '.gif', '.mp4', '.zip')):
                return False
            
            # Skip fragments
            if '#' in url and len(url.split('#')[1]) < 10:
                return False
                
            return True
        except:
            return False

    def extract_comprehensive_content(self, soup, url):
        """Extract ALL content with larger, context-rich chunks"""
        try:
            # Get title with fallbacks
            title = "Untitled"
            for title_selector in ['title', 'h1', '.notion-page-title', '[data-content-editable-leaf="true"]']:
                title_elem = soup.select_one(title_selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)[:300]
                    break
            
            self.logger.info(f"Processing: {title}")
            
            # Remove noise but keep structure
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()
            
            content_sections = []
            
            # Strategy 1: Get the entire main content as one large chunk
            main_selectors = [
                'main', 'article', '.notion-page-content', 
                '[role="main"]', '.notion-selectable', 
                '.notion-page-block', 'body'
            ]
            
            for selector in main_selectors:
                main_elem = soup.select_one(selector)
                if main_elem:
                    # Get all text with some structure
                    main_text = self.get_structured_text(main_elem)
                    if len(main_text) >= MIN_CONTENT_LENGTH:
                        content_sections.append(('full_page', main_text))
                        self.logger.info(f"Extracted full page content: {len(main_text)} chars")
                    break
            
            # Strategy 2: Extract major sections with their subsections
            for h in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                section_content = self.extract_section_with_subsections(h, soup)
                if section_content and len(section_content) >= MIN_CONTENT_LENGTH:
                    content_sections.append(('section', section_content))
            
            # Strategy 3: Extract table/database content
            for table_elem in soup.find_all(['table', '.notion-collection-view', '.notion-database']):
                table_content = self.extract_table_content(table_elem)
                if table_content and len(table_content) >= MIN_CONTENT_LENGTH:
                    content_sections.append(('table', table_content))
            
            # Strategy 4: Extract list content with context
            for list_elem in soup.find_all(['ul', 'ol']):
                list_content = self.extract_list_with_context(list_elem)
                if list_content and len(list_content) >= MIN_CONTENT_LENGTH:
                    content_sections.append(('list', list_content))
            
            # Strategy 5: Extract callouts, quotes, and special blocks
            special_selectors = ['.notion-callout', '.notion-quote', 'blockquote', '.notion-toggle']
            for selector in special_selectors:
                for elem in soup.select(selector):
                    special_content = self.get_structured_text(elem)
                    if special_content and len(special_content) >= MIN_CONTENT_LENGTH:
                        content_sections.append(('special_block', special_content))
            
            # Remove duplicates but keep the longest versions
            unique_sections = self.deduplicate_and_merge_sections(content_sections)
            
            self.logger.info(f"Extracted {len(unique_sections)} content sections from {title}")
            return title, unique_sections
            
        except Exception as e:
            self.logger.error(f"Content extraction failed for {url}: {e}")
            return "Error", []

    def get_structured_text(self, element):
        """Extract text while preserving some structure"""
        if not element:
            return ""
        
        # Replace certain elements with structured text
        for br in element.find_all('br'):
            br.replace_with('\n')
        
        for p in element.find_all('p'):
            if p.get_text(strip=True):
                p.insert_after('\n\n')
        
        for heading in element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(heading.name[1])
            heading_text = heading.get_text(strip=True)
            if heading_text:
                heading.replace_with(f"\n{'#' * level} {heading_text}\n\n")
        
        for li in element.find_all('li'):
            li_text = li.get_text(strip=True)
            if li_text:
                li.replace_with(f"• {li_text}\n")
        
        text = element.get_text()
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()

    def extract_section_with_subsections(self, heading, soup):
        """Extract a section with all its subsections"""
        section_parts = [heading.get_text(strip=True)]
        current_level = int(heading.name[1]) if heading.name.startswith('h') else 1
        
        current = heading.find_next_sibling()
        while current:
            # Stop if we hit a heading of the same or higher level
            if current.name and current.name.startswith('h'):
                next_level = int(current.name[1])
                if next_level <= current_level:
                    break
            
            # Get text content
            if current.name in ['p', 'div', 'ul', 'ol', 'blockquote', 'table']:
                text = self.get_structured_text(current)
                if text and len(text.strip()) > 10:
                    section_parts.append(text)
            
            current = current.find_next_sibling()
            
            # Don't let sections get too long
            if len('\n\n'.join(section_parts)) > MAX_CONTENT_LENGTH * 2:
                break
        
        return '\n\n'.join(section_parts) if len(section_parts) > 1 else None

    def extract_table_content(self, table_elem):
        """Extract table content in a readable format"""
        try:
            rows = []
            
            # Handle regular tables
            if table_elem.name == 'table':
                for row in table_elem.find_all('tr'):
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if any(cells):  # At least one non-empty cell
                        rows.append(' | '.join(cells))
            
            # Handle Notion collection views
            else:
                # Try to extract any structured data
                items = table_elem.find_all(['div', 'span'], class_=re.compile(r'notion|cell|item'))
                for item in items:
                    item_text = item.get_text(strip=True)
                    if item_text and len(item_text) > 5:
                        rows.append(item_text)
            
            if rows:
                return 'Table/Database Content:\n' + '\n'.join(rows)
            return None
            
        except Exception as e:
            self.logger.error(f"Table extraction error: {e}")
            return None

    def extract_list_with_context(self, list_elem):
        """Extract list with surrounding context"""
        # Get preceding context (heading or paragraph)
        context = ""
        prev_elem = list_elem.find_previous_sibling()
        while prev_elem and len(context) < 200:
            if prev_elem.name in ['h1', 'h2', 'h3', 'h4', 'p']:
                prev_text = prev_elem.get_text(strip=True)
                if prev_text:
                    context = prev_text + '\n\n' + context
                    break
            prev_elem = prev_elem.find_previous_sibling()
        
        # Get list items
        items = []
        for li in list_elem.find_all('li'):
            item_text = li.get_text(strip=True)
            if item_text:
                items.append(f"• {item_text}")
        
        if items:
            list_content = '\n'.join(items)
            return context + list_content if context else list_content
        
        return None

    def deduplicate_and_merge_sections(self, sections):
        """Remove duplicates and merge similar content"""
        unique_sections = []
        seen_content = {}
        
        for section_type, content in sections:
            # Create a hash of the first 200 characters for similarity detection
            content_sample = content[:200].lower().strip()
            content_hash = hashlib.md5(content_sample.encode()).hexdigest()
            
            if content_hash in seen_content:
                # If we've seen similar content, keep the longer version
                existing_idx, existing_content = seen_content[content_hash]
                if len(content) > len(existing_content):
                    unique_sections[existing_idx] = (section_type, content)
                    seen_content[content_hash] = (existing_idx, content)
            else:
                unique_sections.append((section_type, content))
                seen_content[content_hash] = (len(unique_sections) - 1, content)
        
        return unique_sections

    def create_large_llm_chunks(self, content_sections, title, url):
        """Create large, context-rich chunks optimized for LLM retrieval"""
        chunks = []
        
        for section_type, content in content_sections:
            # Clean and prepare content
            content = re.sub(r'\s+', ' ', content.strip())
            if len(content) < MIN_CONTENT_LENGTH:
                continue
            
            # Create comprehensive metadata header
            metadata_header = [
                f"=== DOCUMENT METADATA ===",
                f"Title: {title}",
                f"URL: {url}",
                f"Content Type: {section_type}",
                f"Content Length: {len(content)} characters",
                f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"=== CONTENT ===",
                ""
            ]
            
            # For very long content, create overlapping chunks
            if len(content) > MAX_CONTENT_LENGTH:
                # Split into overlapping chunks for better context continuity
                chunk_size = MAX_CONTENT_LENGTH - len('\n'.join(metadata_header)) - 100
                overlap = chunk_size // 4  # 25% overlap
                
                start = 0
                chunk_num = 1
                while start < len(content):
                    end = start + chunk_size
                    chunk_content = content[start:end]
                    
                    # Try to break at sentence boundaries
                    if end < len(content):
                        last_period = chunk_content.rfind('.')
                        last_newline = chunk_content.rfind('\n')
                        break_point = max(last_period, last_newline)
                        
                        if break_point > chunk_size * 0.8:  # Don't break too early
                            chunk_content = chunk_content[:break_point + 1]
                            end = start + len(chunk_content)
                    
                    # Add chunk-specific metadata
                    chunk_metadata = metadata_header + [f"Chunk: {chunk_num} of {section_type}", ""]
                    final_chunk = '\n'.join(chunk_metadata) + chunk_content
                    
                    chunks.append(final_chunk)
                    
                    start = end - overlap
                    chunk_num += 1
                    
                    if start >= len(content) - overlap:
                        break
            else:
                # Single chunk for shorter content
                final_chunk = '\n'.join(metadata_header) + content
                chunks.append(final_chunk)
        
        return chunks

    def generate_embeddings(self, chunks, url, title):
        """Generate embeddings with enhanced metadata"""
        try:
            if not self.model:
                self.logger.info("Loading embedding model...")
                self.model = SentenceTransformer(EMBEDDING_MODEL)
            
            embeddings = []
            
            for i, chunk in enumerate(chunks):
                vector = self.model.encode([chunk])[0]
                
                embedding_data = {
                    "id": f"{hashlib.md5((chunk + url + str(i)).encode()).hexdigest()}",
                    "text": chunk,
                    "url": url,
                    "page_title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "embedding": vector.tolist(),
                    "timestamp": datetime.now().isoformat(),
                    "chunk_length": len(chunk),
                    "word_count": len(chunk.split())
                }
                embeddings.append(embedding_data)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed for {url}: {e}")
            return []

    def scrape_page_with_retry(self, url, depth=0):
        """Scrape a single page with comprehensive retry logic"""
        if url in self.visited_urls or url in self.permanently_failed_urls:
            return []
        
        if depth > MAX_DEPTH:
            self.logger.info(f"Max depth {MAX_DEPTH} reached for {url}")
            return []
        
        retry_count = self.failed_urls.get(url, 0)
        
        for attempt in range(retry_count, MAX_RETRIES + 1):
            if attempt > 0:
                # Calculate and apply retry delay
                delay = self.calculate_retry_delay(attempt)
                self.logger.info(f"Retrying {url} in {delay} seconds (attempt {attempt + 1}/{MAX_RETRIES + 1})")
                time.sleep(delay)
                
                # Restart driver if too many consecutive failures
                if self.consecutive_failures >= DRIVER_RESTART_THRESHOLD:
                    self.logger.info("Too many consecutive failures - restarting driver")
                    self.create_driver()
                    self.consecutive_failures = 0
            
            try:
                self.logger.info(f"Scraping page {len(self.visited_urls) + 1}: {url} (depth: {depth}, attempt: {attempt + 1})")
                
                # Get page content with retry-aware loading
                page_source = self.aggressive_page_load(url, attempt)
                if not page_source:
                    self.logger.warning(f"Failed to load page source for {url} (attempt {attempt + 1})")
                    self.consecutive_failures += 1
                    self.failed_urls[url] = attempt + 1
                    continue  # Try next attempt
                
                # Parse and extract ALL content
                soup = BeautifulSoup(page_source, 'html.parser')
                title, content_sections = self.extract_comprehensive_content(soup, url)
                
                if not content_sections:
                    self.logger.warning(f"No meaningful content found on {url} (attempt {attempt + 1})")
                    # Still try to get links even if no content, but mark as content failure
                    links = self.extract_all_links(soup, url)
                    if links:  # If we got links, consider it a partial success
                        self.visited_urls.add(url)
                        if url in self.failed_urls:
                            del self.failed_urls[url]
                        return links
                    else:
                        self.consecutive_failures += 1
                        self.failed_urls[url] = attempt + 1
                        continue  # Try next attempt
                
                # Create large LLM-optimized chunks
                chunks = self.create_large_llm_chunks(content_sections, title, url)
                
                if chunks:
                    # Generate embeddings
                    embeddings = self.generate_embeddings(chunks, url, title)
                    self.all_embeddings.extend(embeddings)
                    self.logger.info(f"✅ Successfully added {len(embeddings)} embeddings from '{title}' (Total: {len(self.all_embeddings)})")
                
                # Mark as successfully visited
                self.visited_urls.add(url)
                if url in self.failed_urls:
                    del self.failed_urls[url]  # Remove from failed list on success
                
                # Extract ALL possible links for further scraping
                links = self.extract_all_links(soup, url)
                self.consecutive_failures = 0  # Reset on success
                return links
                
            except Exception as e:
                self.logger.error(f"Error scraping {url} (attempt {attempt + 1}): {e}")
                self.consecutive_failures += 1
                self.failed_urls[url] = attempt + 1
                continue  # Try next attempt
        
        # All retries exhausted
        self.logger.error(f"❌ Permanently failed to scrape {url} after {MAX_RETRIES + 1} attempts")
        self.permanently_failed_urls.add(url)
        if url in self.failed_urls:
            del self.failed_urls[url]
        
        return []

    def save_results(self):
        """Save embeddings and comprehensive summary"""
        # Save embeddings
        embeddings_file = os.path.join(OUTPUT_DIR, "deep_embeddings.json")
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_embeddings, f, indent=2, ensure_ascii=False)
        
        # Save detailed summary
        if self.all_embeddings:
            chunk_lengths = [e['chunk_length'] for e in self.all_embeddings]
            word_counts = [e['word_count'] for e in self.all_embeddings]
            
            summary = {
                "scraping_completed": datetime.now().isoformat(),
                "total_embeddings": len(self.all_embeddings),
                "pages_processed": len(self.visited_urls),
                "failed_pages": len(self.permanently_failed_urls),
                "retrying_pages": len(self.failed_urls),
                "unique_pages": list(set(e['url'] for e in self.all_embeddings)),
                "statistics": {
                    "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
                    "max_chunk_length": max(chunk_lengths),
                    "min_chunk_length": min(chunk_lengths),
                    "avg_word_count": sum(word_counts) / len(word_counts),
                    "total_words": sum(word_counts),
                    "total_characters": sum(chunk_lengths)
                },
                "configuration": {
                    "min_content_length": MIN_CONTENT_LENGTH,
                    "max_content_length": MAX_CONTENT_LENGTH,
                    "max_depth": MAX_DEPTH,
                    "embedding_model": EMBEDDING_MODEL
                }
            }
        else:
            summary = {
                "scraping_completed": datetime.now().isoformat(),
                "total_embeddings": 0,
                "pages_processed": len(self.visited_urls),
                "failed_pages": len(self.permanently_failed_urls),
                "retrying_pages": len(self.failed_urls),
                "error": "No embeddings generated"
            }
        
        summary_file = os.path.join(OUTPUT_DIR, "deep_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save visited URLs for debugging
        urls_file = os.path.join(OUTPUT_DIR, "visited_urls.json")
        with open(urls_file, 'w') as f:
            json.dump({
                "visited": list(self.visited_urls),
                "failed": list(self.failed_urls)
            }, f, indent=2)
        
        self.logger.info(f"Saved {len(self.all_embeddings)} embeddings to {embeddings_file}")

    def run(self):
        """Main execution with comprehensive scraping"""
        try:
            self.logger.info(f"Starting DEEP NOTION SCRAPER")
            self.logger.info(f"Target: {START_URL}")
            self.logger.info(f"Configuration: Max depth={MAX_DEPTH}, Chunk size={MIN_CONTENT_LENGTH}-{MAX_CONTENT_LENGTH}")
            self.logger.info("This will scrape EVERY accessible page - may take significant time!")
            
            urls_to_visit = [(START_URL, 0)]
            processed_count = 0
            
            while urls_to_visit:
                url, depth = urls_to_visit.pop(0)
                processed_count += 1
                
                self.logger.info(f"\n--- Processing #{processed_count}: {url} (depth: {depth}) ---")
                self.logger.info(f"Queue: {len(urls_to_visit)} URLs remaining")
                
                # Scrape page and get new links
                new_links = self.scrape_page_with_retry(url, depth)
                
                # Add new links to queue (with deduplication)
                added_links = 0
                for link in new_links:
                    if (link not in self.visited_urls and 
                        link not in self.failed_urls and 
                        not any(link == existing_url for existing_url, _ in urls_to_visit)):
                        urls_to_visit.append((link, depth + 1))
                        added_links += 1
                
                self.logger.info(f"Found {len(new_links)} links, added {added_links} new ones to queue")
                
                # Progress logging
                if processed_count % 5 == 0:
                    self.logger.info(f"\n=== PROGRESS UPDATE ===")
                    self.logger.info(f"Pages processed: {len(self.visited_urls)}")
                    self.logger.info(f"Embeddings created: {len(self.all_embeddings)}")
                    self.logger.info(f"URLs in queue: {len(urls_to_visit)}")
                    self.logger.info(f"Failed pages: {len(self.failed_urls)}")
                
                # Save progress frequently
                if processed_count % 10 == 0:
                    self.save_results()
                    self.logger.info("Progress saved!")
                
                # Be respectful with delays
                time.sleep(2)
            
            # Final save
            self.save_results()
            
            self.logger.info(f"\n=== SCRAPING COMPLETE ===")
            self.logger.info(f"Total pages processed: {len(self.visited_urls)}")
            self.logger.info(f"Total embeddings created: {len(self.all_embeddings)}")
            self.logger.info(f"Failed pages: {len(self.failed_urls)}")
            self.logger.info(f"Average chunk size: {sum(len(e['text']) for e in self.all_embeddings) / len(self.all_embeddings) if self.all_embeddings else 0:.0f} characters")
            
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user - saving progress...")
            self.save_results()
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.save_results()
        finally:
            if self.driver:
                self.driver.quit()

def main():
    scraper = DeepNotionScraper()
    scraper.run()

if __name__ == "__main__":
    main()