# Tip-of-the-Tongue Book Finder

Sometimes we remember a story but forget its title. This project lets us describe the story and suggests books that may match.

## How it works

The search looks for matching words and similar story meanings. It combines both approaches, then reorders the results. The page shows the results from each approach so they can be compared.

Try the [live book finder](https://918xwbrk-8000.eun1.devtunnels.ms/) to find a book from a story description.

## Try it on your computer

From the project folder, install the needed packages and start the app:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The search page opens in your browser. Type a story clue, or choose one of the examples on the page.

## The book data

The project uses the [CMU Book Summary Dataset](https://www.cs.cmu.edu/~dbamman/booksummaries.html), which contains book details and plot summaries. A cleaned copy of 16,356 books is included in this repository.

## The notebook

The notebook compares the search methods on sample questions and measures how long they take. It also saves the top result from each method in a CSV file.
