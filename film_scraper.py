#!/usr/bin/env python3
"""
British Council Films and Festivals Scraper
Extracts information about films and their festival selections/awards
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin
import csv
import argparse
import sys
from pathlib import Path

BASE_URL = "https://filmsandfestivals.britishcouncil.org"
CHECKPOINT_FILE = "scraper_checkpoint.json"

def get_soup(url, timeout=30):
    """Fetch and parse a URL with better error handling"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        time.sleep(0.5)  # Be polite to the server
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_festival_info(text):
    """Extract festival names and status from text"""
    festivals = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        line_lower = line.lower()
        
        # Official Selection pattern
        if 'official selection' in line_lower:
            match = re.search(r'Official Selection\s+(.+?)(?:\s+\d{4})?(?:\s*[-–]\s*(.+?))?$', line, re.IGNORECASE)
            if match:
                festival_name = match.group(1).strip()
                note = match.group(2).strip() if match.group(2) else None
                festivals.append({
                    'festival': festival_name,
                    'status': 'Official Selection',
                    'note': note,
                    'original_text': line
                })
        
        # Winner/Award patterns
        elif 'winner' in line_lower or 'award' in line_lower or 'prize' in line_lower:
            match = re.search(r'(Winner|Award|Prize)\s*[-:]?\s*(.+?)(?:\s+at\s+(.+?))?(?:\s+\d{4})?$', line, re.IGNORECASE)
            if match:
                award_type = match.group(1)
                award_detail = match.group(2).strip() if match.group(2) else ''
                festival_name = match.group(3).strip() if match.group(3) else award_detail
                
                festivals.append({
                    'festival': festival_name,
                    'status': f'{award_type}',
                    'note': award_detail if match.group(3) else None,
                    'original_text': line
                })
            else:
                festivals.append({
                    'festival': line,
                    'status': 'Award/Winner',
                    'note': None,
                    'original_text': line
                })
        
        # Premiere patterns
        elif 'premiere' in line_lower:
            match = re.search(r'(.+?)\s+premiere\s+(?:at\s+)?(.+?)(?:\s+\d{4})?$', line, re.IGNORECASE)
            if match:
                premiere_type = match.group(1).strip()
                festival_name = match.group(2).strip()
                festivals.append({
                    'festival': festival_name,
                    'status': f'{premiere_type} premiere',
                    'note': None,
                    'original_text': line
                })
        
        # Selected for / Screened at patterns
        elif 'selected for' in line_lower or 'screened at' in line_lower:
            match = re.search(r'(?:Selected for|Screened at)\s+(.+?)(?:\s+\d{4})?$', line, re.IGNORECASE)
            if match:
                festival_name = match.group(1).strip()
                festivals.append({
                    'festival': festival_name,
                    'status': 'Selected/Screened',
                    'note': None,
                    'original_text': line
                })
    
    return festivals

def scrape_film_page(url):
    """Scrape individual film page for details"""
    print(f"Scraping: {url}")
    
    soup = get_soup(url)
    if not soup:
        return None
    
    film_data = {
        'url': url,
        'title': None,
        'synopsis': None,
        'year': None,
        'director': None,
        'producer': None,
        'executive_producer': None,
        'writer': None,
        'editor': None,
        'dop': None,
        'composer': None,
        'cast': None,
        'running_time': None,
        'type': None,
        'format': None,
        'genre': None,
        'categories': None,
        'production_status': None,
        'production_company': None,
        'production_company_contact': None,
        'festivals': []
    }
    
    # Get title
    title_elem = soup.find('h1')
    if title_elem:
        film_data['title'] = title_elem.get_text(strip=True)
    
    # Get synopsis - h2 followed by div with text
    synopsis_elem = soup.find('h2', string='Synopsis')
    if synopsis_elem:
        # Find the div that follows the h2
        next_div = synopsis_elem.find_next_sibling('div')
        if next_div:
            film_data['synopsis'] = next_div.get_text(strip=True)
    
    # Get all text content to search for festival mentions
    page_text = soup.get_text()
    film_data['festivals'] = extract_festival_info(page_text)
    
    # Get details from the details section - it's a dl (definition list)
    details_section = soup.find('h2', string='Details')
    if details_section:
        # Find the dl that follows
        dl = details_section.find_next_sibling('dl')
        if dl:
            # Get all dt (term) and dd (definition) pairs
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True).lower().replace(':', '')
                value = dd.get_text(strip=True)
                
                if 'year' in key:
                    film_data['year'] = value
                elif 'type of project' in key or key == 'type':
                    film_data['type'] = value
                elif 'running time' in key:
                    film_data['running_time'] = value
                elif 'format' in key:
                    film_data['format'] = value
                elif 'director' in key and 'photography' not in key:
                    film_data['director'] = value
                elif 'producer' in key and 'executive' not in key:
                    film_data['producer'] = value
                elif 'executive producer' in key:
                    film_data['executive_producer'] = value
                elif 'writer' in key or 'screenplay' in key:
                    film_data['writer'] = value
                elif 'editor' in key:
                    film_data['editor'] = value
                elif 'director of photography' in key or 'cinematographer' in key or key == 'dop':
                    film_data['dop'] = value
                elif 'composer' in key or 'music' in key:
                    film_data['composer'] = value
                elif 'cast' in key or 'principal cast' in key:
                    film_data['cast'] = value
    
    # Get Genre - h2 followed by div with multiple elements
    genre_elem = soup.find('h2', string='Genre')
    if genre_elem:
        next_div = genre_elem.find_next_sibling('div')
        if next_div:
            # Get all text from child elements, not just concatenated text
            genres = []
            for elem in next_div.find_all(['span', 'a', 'p']):
                text = elem.get_text(strip=True)
                if text:
                    genres.append(text)
            
            # If no child elements found, might be direct text with capital letter splits
            if not genres:
                genre_text = next_div.get_text(strip=True)
                # Split on capital letters: "DocumentaryDramaComedy" -> ["Documentary", "Drama", "Comedy"]
                genre_text = re.sub(r'([A-Z][a-z]+)', r' \1', genre_text).strip()
                genres = [g.strip() for g in genre_text.split() if g.strip()]
            
            if genres:
                film_data['genre'] = ', '.join(genres)
    
    # Get Categories - h2 followed by div with multiple elements
    categories_elem = soup.find('h2', string='Categories')
    if categories_elem:
        next_div = categories_elem.find_next_sibling('div')
        if next_div:
            # Get all text from child elements
            categories = []
            for elem in next_div.find_all(['span', 'a', 'p']):
                text = elem.get_text(strip=True)
                if text:
                    categories.append(text)
            
            # If no child elements found, split on capital letters
            if not categories:
                cat_text = next_div.get_text(strip=True)
                cat_text = re.sub(r'([A-Z][a-z]+)', r' \1', cat_text).strip()
                categories = [c.strip() for c in cat_text.split() if c.strip()]
            
            if categories:
                film_data['categories'] = ', '.join(categories)
    
    # Get Production Status - h2 followed by div
    status_elem = soup.find('h2', string='Production Status')
    if status_elem:
        next_div = status_elem.find_next_sibling('div')
        if next_div:
            film_data['production_status'] = next_div.get_text(strip=True)
    
    # Get Production Company - h2 followed by div
    company_elem = soup.find('h2', string='Production Company')
    if company_elem:
        next_div = company_elem.find_next_sibling('div')
        if next_div:
            company_text = next_div.get_text(strip=True)
            # Try to split company name from contact details
            lines = company_text.split('\n')
            if lines:
                film_data['production_company'] = lines[0].strip()
                if len(lines) > 1:
                    film_data['production_company_contact'] = '\n'.join([l.strip() for l in lines[1:] if l.strip()])
    
    return film_data

def get_all_urls_from_file(path):
    if Path(path).exists():
        with open(path, 'r', encoding='utf-8') as f:
            films = json.load(f)
            urls = [f["url"] for f in films if f["url"]]
            return urls

def get_all_film_urls(start_url, max_films=None):
    """Get all film project URLs by paginating through all 1137 pages"""
    print("Fetching film URLs from all pages...")
    film_urls = set()
    
    # The site has 1137 pages with ?page=N parameter
    total_pages = 1137
    page = 1
    
    while page <= total_pages:
        if max_films and len(film_urls) >= max_films:
            break
            
        url = f"{start_url}?page={page}" if page > 1 else start_url
        print(f"Fetching page {page}/{total_pages}... ({len(film_urls)} films found so far)")
        
        soup = get_soup(url)
        if not soup:
            print(f"Failed to fetch page {page}, skipping...")
            page += 1
            continue
        
        # Find all project links on this page
        page_films = 0
        project_links = soup.find_all('a', href=re.compile(r'/projects/[a-z0-9-]+$'))
        for link in project_links:
            href = link.get('href')
            if href and '/projects/' in href and not href.endswith('/projects/'):
                full_url = urljoin(BASE_URL, href)
                if full_url not in film_urls:
                    film_urls.add(full_url)
                    page_films += 1
                    if max_films and len(film_urls) >= max_films:
                        break
        
        if page_films == 0:
            print(f"No films found on page {page}, might be at the end")
            break
            
        page += 1
    
    print(f"\nFound {len(film_urls)} unique film URLs across {page-1} pages")
    return sorted(list(film_urls))

def save_checkpoint(film_urls, processed_urls, all_films):
    """Save current progress to checkpoint file"""
    checkpoint = {
        'film_urls': film_urls,
        'processed_urls': list(processed_urls),
        'all_films': all_films
    }
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    print(f"Checkpoint saved: {len(processed_urls)}/{len(film_urls)} films processed")

def load_checkpoint():
    """Load checkpoint if it exists"""
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_to_json(data, filename='films_data.json'):
    """Save data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved data to {filename}")

def save_to_csv(data, filename='films_data.csv'):
    """Save data to CSV file with flattened festival information"""
    if not data:
        print("No data to save")
        return
    
    # Create rows with one row per film-festival combination
    rows = []
    for film in data:
        if film['festivals']:
            for fest in film['festivals']:
                row = {
                    'title': film['title'],
                    'year': film['year'],
                    'type': film['type'],
                    'director': film['director'],
                    'producer': film['producer'],
                    'executive_producer': film['executive_producer'],
                    'writer': film['writer'],
                    'editor': film['editor'],
                    'dop': film['dop'],
                    'composer': film['composer'],
                    'cast': film['cast'],
                    'running_time': film['running_time'],
                    'format': film['format'],
                    'genre': film['genre'],
                    'categories': film['categories'],
                    'production_status': film['production_status'],
                    'production_company': film['production_company'],
                    'festival': fest['festival'],
                    'festival_status': fest['status'],
                    'festival_note': fest['note'],
                    'festival_original_text': fest.get('original_text', ''),
                    'url': film['url']
                }
                rows.append(row)
        else:
            # Include films with no festival info
            row = {
                'title': film['title'],
                'year': film['year'],
                'type': film['type'],
                'director': film['director'],
                'producer': film['producer'],
                'executive_producer': film['executive_producer'],
                'writer': film['writer'],
                'editor': film['editor'],
                'dop': film['dop'],
                'composer': film['composer'],
                'cast': film['cast'],
                'running_time': film['running_time'],
                'format': film['format'],
                'genre': film['genre'],
                'categories': film['categories'],
                'production_status': film['production_status'],
                'production_company': film['production_company'],
                'festival': None,
                'festival_status': None,
                'festival_note': None,
                'festival_original_text': None,
                'url': film['url']
            }
            rows.append(row)
    
    if rows:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved data to {filename}")

def main():
    """Main scraping function"""
    parser = argparse.ArgumentParser(description='Scrape British Council Films and Festivals database')
    parser.add_argument('--max', type=int, help='Maximum number of films to scrape')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--output', default='films_data', help='Output filename prefix (without extension)')
    args = parser.parse_args()
    
    start_url = "https://filmsandfestivals.britishcouncil.org/"
    
    # Check for resume
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            print(f"Resuming from checkpoint: {len(checkpoint['processed_urls'])}/{len(checkpoint['film_urls'])} films already processed")
            film_urls = checkpoint['film_urls']
            processed_urls = set(checkpoint['processed_urls'])
            all_films = checkpoint['all_films']
        else:
            print("No checkpoint found, starting fresh")
            args.resume = False
    
    if not args.resume:
        # Get all film URLs
        film_urls = get_all_urls_from_file("films_data.json")
        if not film_urls:
            print("No film URLs found. Exiting.")
            return
        processed_urls = set()
        all_films = []
    
    # Scrape each film
    try:
        for i, url in enumerate(film_urls, 1):
            if url in processed_urls:
                continue
            
            print(f"\nProcessing {len(processed_urls)+1}/{len(film_urls)}")
            try:
                film_data = scrape_film_page(url)
                if film_data:
                    all_films.append(film_data)
                    processed_urls.add(url)
                else:
                    print(f"Failed to scrape {url}")
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving checkpoint...")
                save_checkpoint(film_urls, processed_urls, all_films)
                raise
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
            
            # Save checkpoint every 10 films
            if len(processed_urls) % 10 == 0:
                save_checkpoint(film_urls, processed_urls, all_films)
    
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(1)
    
    # Save final results
    save_to_json(all_films, f'{args.output}.json')
    save_to_csv(all_films, f'{args.output}.csv')
    
    # Clean up checkpoint
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Scraping complete!")
    print(f"Total films scraped: {len(all_films)}")
    
    films_with_festivals = sum(1 for f in all_films if f['festivals'])
    print(f"Films with festival info: {films_with_festivals}")
    
    total_festival_mentions = sum(len(f['festivals']) for f in all_films)
    print(f"Total festival mentions: {total_festival_mentions}")
    
    # Count unique festivals
    all_festival_names = set()
    for film in all_films:
        for fest in film['festivals']:
            all_festival_names.add(fest['festival'])
    print(f"Unique festivals mentioned: {len(all_festival_names)}")
    
    # Show some examples
    print(f"\n{'='*60}")
    print("Examples of films with festival information:")
    count = 0
    for film in all_films:
        if film['festivals'] and count < 5:
            print(f"\n{film['title']} ({film['year']})")
            for fest in film['festivals'][:3]:
                print(f"  - {fest['status']}: {fest['festival']}")
                if fest['note']:
                    print(f"    Note: {fest['note']}")
            count += 1

if __name__ == "__main__":
    main()