// In serverless functions, we need to use the absolute URL
const getBaseUrl = () => {
  if (typeof window !== "undefined") {
    return window.location.origin;
  }
  
  // For Vercel serverless functions
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`;
  }
  
  // For local development
  return process.env.NEXT_PUBLIC_BASE_URL || "http://localhost:3000";
};

export async function log(message: string, type: "crawler" | "ai_summary" | "ai_analysis") {
  try {
    // Create a specific timestamp in ISO format
    const now = new Date().toISOString();
    
    // When running in serverless function, we need to make a direct DB call instead of an HTTP request
    if (typeof window === "undefined") {
      const { neon } = await import("@neondatabase/serverless");
      const sql = neon(process.env.DATABASE_URL!);
      await sql`INSERT INTO logs (timestamp, message, type) VALUES (${now}, ${message}, ${type})`;
      return;
    }

    // Client-side logging via API
    const baseUrl = getBaseUrl();
    await fetch(`${baseUrl}/api/logs`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message, type }),
    });
  } catch (error) {
    console.error("Failed to save log:", error);
  }
}

