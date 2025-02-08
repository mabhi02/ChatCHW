"use client"

import { useEffect, useState, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { Activity, Clock, FileText, RefreshCcw, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { ScrollArea } from "@/components/ui/scroll-area"

interface CronStats {
  total_articles: number
  last_crawl: string
  articles_last_hour: number
  articles_last_day: number
}

interface CrawlEntry {
  updated_at: string
  articles_processed: number
}

interface CronStatus {
  stats: CronStats
  recentCrawls: CrawlEntry[]
}

export function CronMonitor() {
  const [status, setStatus] = useState<CronStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [crawling, setCrawling] = useState(false)
  const { toast } = useToast()

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      console.log("Fetching cron status...")
      const response = await fetch("/api/cron/status")
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Failed to fetch status: ${response.status} ${errorText}`)
      }
      const data = await response.json()
      console.log("Cron status fetched successfully:", data)
      setStatus(data)
    } catch (err) {
      console.error("Error fetching cron status:", err)
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(`Failed to load cron status: ${errorMessage}`)
      toast({
        title: "Error",
        description: `Failed to load cron status: ${errorMessage}`,
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }, [toast])

  const triggerCrawl = useCallback(async () => {
    if (crawling) return
    
    try {
      setCrawling(true)
      setError(null)
      console.log("Triggering manual crawl...")
      const response = await fetch("/api/cron/crawl", { method: "POST" })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(
          `Failed to trigger crawl: ${response.status} ${response.statusText}\n${JSON.stringify(result, null, 2)}`,
        )
      }

      console.log("Manual crawl completed:", result)
      if (result.success) {
        toast({
          title: "Crawl Job Completed",
          description: `Processed ${result.result.totalProcessed} articles.`,
        })
        await fetchStatus()
      } else {
        throw new Error(result.error || "Unknown error occurred")
      }
    } catch (err) {
      console.error("Error triggering crawl:", err)
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(`Failed to trigger crawl: ${errorMessage}`)
      toast({
        title: "Error",
        description: `Failed to trigger crawl: ${errorMessage}`,
        variant: "destructive",
      })
    } finally {
      setCrawling(false)
    }
  }, [crawling, toast, fetchStatus])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 60000)
    return () => clearInterval(interval)
  }, [fetchStatus])

  if (loading && !status) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    )
  }

  if (!status) return null

  const lastCrawlTime = new Date(status.stats.last_crawl)
  const timeSinceLastCrawl = Math.floor((Date.now() - lastCrawlTime.getTime()) / 1000 / 60)
  const isHealthy = timeSinceLastCrawl <= 20

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Articles</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{status.stats.total_articles}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Hour</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{status.stats.articles_last_hour}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last 24 Hours</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{status.stats.articles_last_day}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cron Status</CardTitle>
            <RefreshCcw className={`h-4 w-4 ${isHealthy ? "text-green-500" : "text-red-500"}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{isHealthy ? "Healthy" : "Late"}</div>
            <p className="text-xs text-muted-foreground">Last run: {timeSinceLastCrawl} minutes ago</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Recent Crawls</CardTitle>
          <Button onClick={triggerCrawl} disabled={crawling}>
            <Play className="mr-2 h-4 w-4" />
            {crawling ? "Crawling..." : "Start Crawl"}
          </Button>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[300px] pr-4">
            <div className="space-y-2">
              {status.recentCrawls.map((crawl, index) => (
                <div key={index} className="flex items-center justify-between border-b pb-2">
                  <div>
                    <p className="font-medium">{new Date(crawl.updated_at).toLocaleString()}</p>
                    <p className="text-sm text-muted-foreground">{crawl.articles_processed} articles processed</p>
                  </div>
                  <Activity
                    className={`h-4 w-4 ${crawl.articles_processed > 0 ? "text-green-500" : "text-yellow-500"}`}
                  />
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}

