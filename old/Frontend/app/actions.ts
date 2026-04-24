"use server"

import { PrismaClient } from '@prisma/client'
import { z } from "zod"

const prisma = new PrismaClient()

const articleSchema = z.object({
  id: z.number(),
  title: z.string(),
  url: z.string().url(),
  publishedAt: z.date().transform(date => date.toISOString()),
  themes: z.array(z.string()),
  insights: z.array(z.string()),
  predictions: z.array(z.string()),
  fullText: z.string(),
  updated_at: z.date().transform(date => date.toISOString()),
  summary: z.string(),
  topics: z.array(z.string()),
})

type Article = z.infer<typeof articleSchema>

export async function getLatestArticles(limit = 10) {
  console.log("getLatestArticles called with limit:", limit)
  try {
    const articles = await prisma.article.findMany({
      take: limit,
      orderBy: {
        publishedAt: 'desc'
      },
      select: {
        id: true,
        title: true,
        url: true,
        publishedAt: true,
        themes: true,
        insights: true,
        predictions: true,
        fullText: true,
        updatedAt: true,
        summary: true,
        topics: true
      }
    })

    const formattedArticles = articles.map(article => ({
      ...article,
      id: article.id,
      updated_at: article.updatedAt,
      summary: article.summary || '',
      topics: article.topics || []
    }))

    console.log(`Retrieved ${articles.length} articles`)
    return articleSchema.array().parse(formattedArticles)
  } catch (error) {
    console.error("Failed to fetch articles:", error)
    throw new Error("Failed to fetch articles")
  }
}

export async function saveArticle(article: Article) {
  const validatedArticle = articleSchema.parse(article)

  try {
    await prisma.article.upsert({
      where: {
        url: validatedArticle.url
      },
      create: {
        title: validatedArticle.title,
        url: validatedArticle.url,
        publishedAt: new Date(validatedArticle.publishedAt),
        themes: validatedArticle.themes,
        insights: validatedArticle.insights,
        predictions: validatedArticle.predictions,
        fullText: validatedArticle.fullText,
        summary: validatedArticle.summary || '',
        topics: validatedArticle.topics || []
      },
      update: {
        title: validatedArticle.title,
        publishedAt: new Date(validatedArticle.publishedAt),
        themes: validatedArticle.themes,
        insights: validatedArticle.insights,
        predictions: validatedArticle.predictions,
        fullText: validatedArticle.fullText,
        summary: validatedArticle.summary || '',
        topics: validatedArticle.topics || [],
        updatedAt: new Date()
      }
    })
    return { success: true }
  } catch (error) {
    console.error("Failed to save article:", error)
    return { success: false, error: "Failed to save article" }
  }
}

export async function getArticleById(id: string) {
  try {
    const article = await prisma.article.findUnique({
      where: {
        id: parseInt(id)
      }
    })

    if (!article) {
      return null
    }

    return {
      ...article,
      id: article.id,
      updated_at: article.updatedAt,
      summary: article.summary || '',
      topics: article.topics || []
    }
  } catch (error) {
    console.error("Failed to fetch article:", error)
    return null
  }
}

