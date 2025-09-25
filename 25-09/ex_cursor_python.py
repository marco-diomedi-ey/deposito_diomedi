import ddgs

#scrivi una funzione che implementi una somma tra due numeri
def somma(a, b):
    """Calculates the sum of two numbers.
    
    Args:
        a (int or float): The first number to add.
        b (int or float): The second number to add.
    
    Returns:
        int or float: The sum of a and b.
    """
    return a + b

# Slugify function to convert a text to a slug
def slugify(text):
    """Converts text to a URL-friendly slug format.
    
    Args:
        text (str): The input text to convert to a slug.
    
    Returns:
        str: The slugified text with lowercase letters and hyphens instead of spaces.
    """
    return text.lower().replace(" ", "-")

#function for internet search using ddgs and return the results
def search_internet(query):
    """Searches the internet for a query using ddgs and returns the results.
    
    Args:
        query (str): The search query to use.
    
    Returns:
        Any: The results of the search, depending on the underlying ddgs API.
    """
    # Support multiple ddgs/duckduckgo_search APIs
    # 1) Function-style APIs
    if hasattr(ddgs, "ddg") and callable(getattr(ddgs, "ddg")):
        return ddgs.ddg(query, verify=False)
    if hasattr(ddgs, "ddgs") and callable(getattr(ddgs, "ddgs")):
        return ddgs.ddgs(query, verify=False)
    if hasattr(ddgs, "search") and callable(getattr(ddgs, "search")):
        return ddgs.search(query, verify=False)
    if hasattr(ddgs, "text") and callable(getattr(ddgs, "text")):
        return ddgs.text(query, verify=False)

    # 2) Class-based API from duckduckgo_search
    if hasattr(ddgs, "DDGS"):
        try:
            client = ddgs.DDGS(verify=False)
            if hasattr(client, "text") and callable(getattr(client, "text")):
                return list(client.text(query))
            if hasattr(client, "search") and callable(getattr(client, "search")):
                return list(client.search(query))
        except Exception:
            raise

    raise AttributeError("Unsupported ddgs API: expected functions ddg/ddgs/search/text or class DDGS with .text/.search")

# Main function
if __name__ == "__main__":
    print(somma(1, 2))
    print(slugify("Hello World"))
    print(search_internet("Hello World"))