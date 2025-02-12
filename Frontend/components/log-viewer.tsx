"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import { ScrollArea } from "@/components/ui/scroll-area"

interface Log {
  timestamp: string
  message: string
  type: "crawler" | "ai_summary" | "ai_analysis"
}

export function LogViewer() {
  const [logs, setLogs] = useState<Log[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchLogs() {
      try {
        const response = await fetch("/api/logs")
        if (!response.ok) throw new Error("Failed to fetch logs")
        const data = await response.json()
        setLogs(data)
      } catch (err) {
        setError("Failed to load logs")
      } finally {
        setLoading(false)
      }
    }

    fetchLogs()
    const interval = setInterval(fetchLogs, 60000) // Refresh every minute
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="space-y-4">
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

  const crawlerLogs = logs.filter((log) => log.type === "crawler")
  const aiSummaryLogs = logs.filter((log) => log.type === "ai_summary")
  const aiAnalysisLogs = logs.filter((log) => log.type === "ai_analysis")

  return (
    <Card>
      <CardHeader>
        <CardTitle>System Logs</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="crawler">
          <TabsList>
            <TabsTrigger value="crawler">Crawler</TabsTrigger>
            <TabsTrigger value="ai_summary">AI Summary</TabsTrigger>
            <TabsTrigger value="ai_analysis">AI Analysis</TabsTrigger>
          </TabsList>
          <TabsContent value="crawler">
            <LogList logs={crawlerLogs} />
          </TabsContent>
          <TabsContent value="ai_summary">
            <LogList logs={aiSummaryLogs} />
          </TabsContent>
          <TabsContent value="ai_analysis">
            <LogList logs={aiAnalysisLogs} />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

function LogList({ logs }: { logs: Log[] }) {
  return (
    <ScrollArea className="h-[300px]">
      <div className="space-y-1 pr-4">
        {logs.map((log, index) => (
          <div key={index} className="text-sm border-b border-border/50 pb-1">
            <span className="font-mono text-muted-foreground text-xs">{new Date(log.timestamp).toLocaleString()}</span>
            <span className="ml-2">{log.message}</span>
          </div>
        ))}
      </div>
    </ScrollArea>
  )
}

