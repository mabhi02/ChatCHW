import { neon } from "@neondatabase/serverless"
import dotenv from "dotenv"

dotenv.config()

const sql = neon(process.env.DATABASE_URL)

async function updateSchema() {
  try {
    // Drop existing tables if they exist
    await sql`DROP TABLE IF EXISTS articles CASCADE`
    await sql`DROP TABLE IF EXISTS logs CASCADE`

    // Create the articles table
    await sql`
      CREATE TABLE articles (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        url TEXT UNIQUE NOT NULL,
        published_at TIMESTAMP WITH TIME ZONE NOT NULL,
        themes TEXT[] NOT NULL,
        insights TEXT[] NOT NULL,
        predictions TEXT[] NOT NULL,
        full_text TEXT NOT NULL,
        summary TEXT,
        topics TEXT[],
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
      )
    `

    // Create the logs table
    await sql`
      CREATE TABLE logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        message TEXT NOT NULL,
        type TEXT NOT NULL
      )
    `

    console.log("Schema recreated successfully")
  } catch (error) {
    console.error("Error recreating schema:", error)
  }
}

updateSchema()

