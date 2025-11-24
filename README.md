# British Council Films and Festivals Scraper

This scraper extracts information about films and their festival selections/awards from the British Council Films and Festivals database (https://filmsandfestivals.britishcouncil.org/).

## Features

- Scrapes all film project pages from the database
- Extracts comprehensive film information (title, director, cast, crew, etc.)
- Identifies festival selections, awards, and premieres
- Saves data in both JSON and CSV formats
- Checkpoint/resume functionality for interrupted scraping
- Progress tracking and error handling

## Requirements

```bash
pip install requests beautifulsoup4
```

## Usage

### Basic usage (scrape all films):
```bash
python festival_scraper.py
```

## Output Files

### JSON Output (`films_data.json`)
Contains complete film information including:
- Film metadata (title, year, type, director, producer, etc.)
- Full crew details (editor, DOP, composer, cast)
- Production information (company, status, genre, categories)
- Array of festival information with status and notes

### CSV Output (`films_data.csv`)
Flattened format with one row per film-festival combination:
- All film metadata columns
- Festival name, status (selection/award/premiere), and notes
- Original festival mention text for reference

## Extracted Information

### Film Details
- Title
- Synopsis
- Year
- Type (Feature, Short, Documentary, etc.)
- Running time
- Format (DCP, etc.)
- Director
- Producer
- Executive Producer
- Writer
- Editor
- Director of Photography
- Composer
- Principal Cast
- Genre
- Categories
- Production Status
- Production Company (with contact details)

### Festival Information
The scraper identifies:
- **Official Selections** (e.g., "Official Selection Sundance 2024")
- **Awards and Winners** (e.g., "Winner - Best Documentary")
- **Premieres** (e.g., "World premiere", "European premiere")
- **Screenings** (e.g., "Selected for", "Screened at")

Each festival mention includes:
- Festival name
- Status/type (Official Selection, Winner, Award, Premiere, etc.)
- Additional notes (e.g., "World premiere", "Best Documentary")
- Original text for verification

## Notes

- The scraper includes a 0.5 second delay between requests to be respectful to the server
- Festival information extraction uses pattern matching on the page text
- Some festival mentions may need manual review for accuracy
- The scraper saves raw original text for each festival mention for verification


### Using the JSON output:
- Full nested data structure for programmatic access
- Easier to work with for detailed film information
- Better for database import or further processing

## License

This scraper is for educational and research purposes. Please respect the British Council's terms of service and copyright.
