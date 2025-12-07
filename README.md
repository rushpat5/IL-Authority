# Internal Authority Flow Analyzer 🌿

This tool helps you understand **how internal links move authority around your website**.

Backlinks give your site “power,” but internal links decide **where that power actually goes**.  
This app shows you which pages receive the most and least internal authority so you can improve your structure.

---

## How it works (super simple)

1. **Upload a CSV** of your internal links  
   Your file should contain two columns:
   - `source_url` → the page that has a link  
   - `target_url` → the page it links to  

2. The app:
   - Cleans and normalizes URLs  
   - Removes navigation/footer links (these appear everywhere and distort results)  
   - Builds a map of how your pages link together  
   - Uses a PageRank-style model to simulate how authority flows  
   - Shows which pages get stronger or weaker  

3. You get:
   - A list of your strongest pages  
   - A list of your weakest pages  
   - A simple bar chart  
   - Suggestions based on your structure  

---

## Why this matters

Backlinks often point to only a few pages.  
Your internal links decide whether that value spreads to the pages you *want* to rank.

If your important pages are weak internally, this tool will show you exactly that.

---

## Input Format

Example:

```csv
source_url,target_url
https://example.com/,https://example.com/blog/
https://example.com/blog/,https://example.com/post-1/
https://example.com/post-1/,https://example.com/product-a/
