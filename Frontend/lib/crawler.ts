import { neon } from "@neondatabase/serverless"
import { JSDOM } from "jsdom"
import { generateSummaryAndTopics, generateNewsAnalysis } from "./ai-utils"
import type { NewsArticle } from "../types/news-article"
import { log } from "./logger"

const sql = neon(process.env.DATABASE_URL!)

interface PageResult {
  links: string[]
  hasNextPage: boolean
}

async function getArticleLinksFromPage(pageUrl: string): Promise<PageResult> {
  const response = await fetch(pageUrl)
  if (!response.ok) {
    throw new Error(`Failed to fetch page ${pageUrl}: ${response.status} ${response.statusText}`)
  }

  const html = await response.text()
  const dom = new JSDOM(html)
  const document = dom.window.document

  const articleLinks = Array.from(document.querySelectorAll<HTMLAnchorElement>("h2 a")).map((a) => a.href)
  const hasNextPage = !!document.querySelector('.wp-block-query-pagination-next')

  return {
    links: articleLinks,
    hasNextPage
  }
}

async function processArticle(articleUrl: string, existingArticleUrls: Set<string>): Promise<NewsArticle | null> {
  if (existingArticleUrls.has(articleUrl)) {
    console.log(`Article already in database: ${articleUrl}`)
    return null
  }

  console.log(`Processing article: ${articleUrl}`)
  await log(`Processing article: ${articleUrl}`, "crawler")

  const articleResponse = await fetch(articleUrl)
  if (!articleResponse.ok) {
    throw new Error(`Failed to fetch article ${articleUrl}: ${articleResponse.status} ${articleResponse.statusText}`)
  }

  const articleHtml = await articleResponse.text()
  const articleDom = new JSDOM(articleHtml)
  const articleDoc = articleDom.window.document

  const title = articleDoc.querySelector("h1")?.textContent?.trim()
  const dateStr = articleDoc.querySelector("time")?.getAttribute("datetime")
  const fullText = articleDoc.querySelector("main")?.textContent?.trim()

  if (!title || !dateStr || !fullText) {
    throw new Error(`Missing required data for article ${articleUrl}`)
  }

  console.log(`Generating initial analysis for article: ${title}`)
  const articleDetails = await generateSummaryAndTopics({
    title,
    url: articleUrl,
    dateStr,
    fullText
  })

  console.log(`Generating predictions for article: ${title}`)
  const { predictions } = await generateNewsAnalysis(articleDetails)

  const article: NewsArticle = {
    ...articleDetails,
    predictions
  }

  await sql`
    INSERT INTO articles (
      title, url, published_at, themes, insights, predictions, full_text, summary, topics
    ) VALUES (
      ${article.title},
      ${article.url},
      ${article.publishedAt},
      ${article.themes},
      ${article.insights},
      ${article.predictions},
      ${article.fullText},
      ${article.summary},
      ${article.topics}
    )
    ON CONFLICT (url) DO UPDATE SET
      title = EXCLUDED.title,
      published_at = EXCLUDED.published_at,
      themes = EXCLUDED.themes,
      insights = EXCLUDED.insights,
      predictions = EXCLUDED.predictions,
      full_text = EXCLUDED.full_text,
      summary = EXCLUDED.summary,
      topics = EXCLUDED.topics,
      updated_at = CURRENT_TIMESTAMP
  `

  console.log(`Processed and saved article: ${article.title}`)
  await log(`Processed and saved article: ${article.title}`, "crawler")
  return article
}

export async function crawlWhiteHouseNews() {
  const baseUrl = "https://www.whitehouse.gov/news/"
  const processedUrls = new Set()
  const newArticles: NewsArticle[] = []
  let currentPage = 1
  const BATCH_SIZE = 5
  const MAX_ARTICLES = 20

  try {
    console.log("Starting White House news crawler")
    await log("Starting White House news crawler", "crawler")

    console.log("Fetching all existing articles from database")
    const existingArticles = await sql`
      SELECT url FROM articles
    `
    const existingArticleUrls = new Set(existingArticles.map((a) => a.url))
    console.log(`Found ${existingArticleUrls.size} existing articles in database`)
    await log(`Found ${existingArticleUrls.size} existing articles in database`, "crawler")

    let hasMorePages = true
    let consecutiveEmptyPages = 0
    const MAX_EMPTY_PAGES = 3

    while (hasMorePages && consecutiveEmptyPages < MAX_EMPTY_PAGES && newArticles.length < MAX_ARTICLES) {
      const pageUrl = currentPage === 1 ? baseUrl : `${baseUrl}page/${currentPage}/`
      console.log(`Fetching news page ${currentPage}: ${pageUrl}`)
      await log(`Fetching news page ${currentPage}`, "crawler")

      try {
        const { links: articleLinks, hasNextPage } = await getArticleLinksFromPage(pageUrl)
        hasMorePages = hasNextPage
        
        console.log(`Found ${articleLinks.length} articles on page ${currentPage}`)
        await log(`Found ${articleLinks.length} articles on page ${currentPage}`, "crawler")

        let newArticlesInPage = 0
        for (let i = 0; i < articleLinks.length && newArticles.length < MAX_ARTICLES; i += BATCH_SIZE) {
          const batch = articleLinks.slice(i, i + BATCH_SIZE)
          const batchResults = await Promise.all(
            batch.map(async (url: string) => {
              if (processedUrls.has(url)) return null
              processedUrls.add(url)
              
              try {
                return await processArticle(url, existingArticleUrls)
              } catch (error) {
                console.error(`Error processing article ${url}:`, error)
                await log(`Error processing article ${url}: ${error}`, "crawler")
                return null
              }
            })
          )

          const validResults = batchResults.filter((article: NewsArticle | null): article is NewsArticle => article !== null)
          newArticles.push(...validResults)
          newArticlesInPage += validResults.length

          console.log(`Processed batch of ${batch.length} articles, got ${validResults.length} new articles`)
          await log(`Processed batch of ${batch.length} articles, got ${validResults.length} new articles`, "crawler")
        }

        if (newArticlesInPage === 0) {
          consecutiveEmptyPages++
          console.log(`No new articles found on page ${currentPage}. Empty pages: ${consecutiveEmptyPages}/${MAX_EMPTY_PAGES}`)
          await log(`No new articles found on page ${currentPage}. Empty pages: ${consecutiveEmptyPages}/${MAX_EMPTY_PAGES}`, "crawler")
        } else {
          consecutiveEmptyPages = 0
        }

        currentPage++
      } catch (error) {
        console.error(`Error processing page ${currentPage}:`, error)
        await log(`Error processing page ${currentPage}: ${error}`, "crawler")
        hasMorePages = false
      }
    }

    const stopReason = newArticles.length >= MAX_ARTICLES 
      ? `reached maximum article limit of ${MAX_ARTICLES}`
      : consecutiveEmptyPages >= MAX_EMPTY_PAGES 
        ? `hit ${MAX_EMPTY_PAGES} consecutive pages with no new articles`
        : 'reached last page'
    
    console.log(`Crawler finished (${stopReason}). Processed ${newArticles.length} new articles across ${currentPage - 1} pages.`)
    await log(`Crawler finished (${stopReason}). Processed ${newArticles.length} new articles across ${currentPage - 1} pages.`, "crawler")
    return { newArticles, totalProcessed: processedUrls.size }
  } catch (error) {
    console.error(`Crawler failed:`, error)
    await log(`Crawler failed: ${error}`, "crawler")
    throw error
  }
}

