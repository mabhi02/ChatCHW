import { generateObject } from "ai"
import { openai } from "@ai-sdk/openai"
import { z } from "zod"
import type { NewsArticle } from "../types/news-article"
import { log } from "./logger"

// Rate limiter class for OpenAI API calls
class RateLimiter {
  private lastCallTimes: Map<string, number> = new Map()
  private modelIntervals: { [key: string]: number } = {
    'gpt-4o': 6000, // 10 requests per minute
    'o1': 60000,      // 1 request per minute
  }

  async waitForNextCall(model: string): Promise<void> {
    const now = Date.now()
    const minInterval = this.modelIntervals[model] || 60000 // Default to 1 per minute if model not specified
    const lastCallTime = this.lastCallTimes.get(model) || 0
    const timeSinceLastCall = now - lastCallTime
    
    if (timeSinceLastCall < minInterval) {
      const waitTime = minInterval - timeSinceLastCall
      console.log(`Rate limiting ${model}: waiting ${waitTime}ms before next API call`)
      await new Promise(resolve => setTimeout(resolve, waitTime))
    }
    
    this.lastCallTimes.set(model, Date.now())
  }
}

const rateLimiter = new RateLimiter()

const summarySchema = z.object({
  title: z.string(),
  url: z.string().url(),
  publishedAt: z.string().transform((date) => {
    // Ensure the date is in a valid format
    const parsed = new Date(date);
    if (isNaN(parsed.getTime())) {
      throw new Error("Invalid date format");
    }
    return parsed.toISOString();
  }),
  fullText: z.string(),
  summary: z.string(),
  topics: z.array(z.string()),
  themes: z.array(z.string()),
  insights: z.array(z.string()),
})

const newsAnalysisSchema = z.object({
  predictions: z.array(z.string()),
})

export async function generateSummaryAndTopics(
  article: { title: string; url: string; dateStr: string; fullText: string }
): Promise<Omit<NewsArticle, 'predictions'>> {
  try {
    await log(`Generating summary and topics for article: ${article.title}`, "ai_summary")

    const model = "gpt-4o"
    await rateLimiter.waitForNextCall(model)
    const { object } = await generateObject({
      model: openai(model),
      schema: summarySchema,
      prompt: `Analyze this White House news article and extract key information:
      
      Title: ${article.title}
      URL: ${article.url}
      Date: ${article.dateStr}
      Content: ${article.fullText}
      
      Provide a comprehensive analysis including:
      - A concise summary (2-3 sentences)
      - 3-5 main topics
      - Key themes from the article
      - Important insights and implications
      
      Format the response according to the schema, ensuring all fields are populated.`
    })

    await log(`Generated summary and topics for article: ${article.title}`, "ai_summary")
    return {
      ...object,
      url: article.url,
      publishedAt: article.dateStr,
      fullText: article.fullText
    }
  } catch (error) {
    await log(`Error generating summary and topics for article: ${article.title}. Error: ${error}`, "ai_summary")
    console.error("Error generating summary and topics:", error)
    throw error
  }
}

export async function generateNewsAnalysis(
  article: Omit<NewsArticle, 'predictions'>
): Promise<Pick<NewsArticle, 'predictions'>> {
  try {
    await log(`Generating predictions for article: ${article.title}`, "ai_analysis")
    
    const model = "o1"
    await rateLimiter.waitForNextCall(model)
    const { object } = await generateObject({
      model: openai(model),
      schema: newsAnalysisSchema,
      prompt: `Based on this analyzed White House news article, generate predictions about potential future developments and implications:
      
      Title: ${article.title}
      Summary: ${article.summary}
      Topics: ${article.topics.join(', ')}
      Themes: ${article.themes.join(', ')}
      Insights: ${article.insights.join(', ')}
      
      Provide 3-5 specific, well-reasoned predictions about potential future developments, policy impacts, or follow-up actions based on this news.`
    })

    await log(`Generated predictions for article: ${article.title}`, "ai_analysis")
    return object
  } catch (error) {
    await log(`Error generating predictions for article: ${article.title}. Error: ${error}`, "ai_analysis")
    console.error("Error generating predictions:", error)
    throw error
  }
}

