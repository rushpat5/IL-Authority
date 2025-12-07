# Internal Authority Flow Analyzer 🕸️

A minimal Streamlit app that helps you understand how **link authority flows through your site** via internal links.

Backlinks light up specific pages, but your **internal linking structure** decides where that value actually goes. This tool models that flow using a **PageRank-style algorithm** on your internal link graph.

---

## What this tool does

Given a CSV of internal links, it will:

1. **Load your internal links**  
   Expects columns:
   - `source_url`
   - `target_url`

2. **Normalize URLs (optional)**  
   - Strip query parameters (`?utm_source=...`)
   - Strip fragments (`#section`)
   - Normalize trailing slashes

3. **Remove boilerplate navigation/footer links**  
   - Identifies targets linked from many pages (e.g. > 80% of pages)
   - Treats those as nav/footer and removes them from the graph  
   - This leaves mostly **editorial links** (body/contextual links)

4. **Build the internal link graph**  
   - Each URL becomes a node  
   - Each internal link becomes a directed edge `source_url → target_url`

5. **Run a PageRank-style authority model**  
   - Uses a **damping factor** (default: 0.85, i.e. about 15% loss per hop)  
   - Computes an `authority_score` per URL  
   - Normalizes this to an `authority_index` between 0 and 100

6. **Visualize and inspect results**  
   - Bar chart of top pages by internal authority  
   - Full table of all pages with:
     - `url`
     - `authority_score` (raw PageRank)
     - `authority_index` (0–100 scale)

---

## Why this is useful

- Shows **which pages accumulate internal authority**
- Highlights **“authority sinks”** and **under-connected money pages**
- Helps you design better **internal link structures and hub pages**
- Lets you **squeeze more value from existing backlinks** without buying new ones

It does **not** crawl your site or read content.  
It only uses the **structure of internal links** (which page links to which page).

---

## Input format

The app expects a CSV file with at least:

```csv
source_url,target_url
https://example.com/,https://example.com/blog/
https://example.com/,https://example.com/contact/
https://example.com/blog/,https://example.com/post-a/
https://example.com/blog/,https://example.com/post-b/
https://example.com/post-a/,https://example.com/post-b/
https://example.com/post-b/,https://example.com/post-c/
https://example.com/post-c/,https://example.com/post-a/
https://example.com/blog/,https://example.com/
